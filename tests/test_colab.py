"""The Colab driver against a fake runtime: verified sync, resume and when the runtime stops."""

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from crisisline import colab
from crisisline.colab import FILE_MARKER, JOB_MARKER, ColabError, Job, Snapshot, Status

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = "artifacts/generations/m.jsonl"
MANIFEST = "artifacts/generations/m.manifest.json"


@pytest.fixture
def cfg() -> dict[str, Any]:
    with (REPO_ROOT / "configs" / "colab.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def job() -> Job:
    return Job(
        session="crisisline-m-main",
        commit="a" * 40,
        tag="qwen3.5:4b-q8_0",
        ollama_version="",
        generate_args=("--model", "m", "--limit", "10"),
        outputs=(OUT, MANIFEST),
    )


class FakeRuntime(colab.Colab):
    """A runtime whose generate appends ``per_poll`` records at every status call.

    ``fail[call]`` makes the next calls of ``new``, ``setup``, ``launch`` or ``stop`` raise. A
    failing ``new`` or ``launch`` raises after doing its work (a hang whose command went
    through); ``lost_launches`` launches raise without launching.
    """

    def __init__(
        self, files: dict[str, bytes] | None = None, per_poll: int = 2, polls: int = 2
    ) -> None:
        super().__init__("crisisline-m-main", 60)
        self.files = dict(files or {})  # repository-relative path -> content
        self.up = bool(files)
        self.per_poll, self.polls_left = per_poll, polls
        self.running = False
        self.corrupt_downloads = 0
        self.fail: dict[str, int] = {}
        self.lost_launches = 0
        self.calls: list[str] = []
        self.scripts: list[str] = []

    def _maybe_fail(self, call: str) -> None:
        if self.fail.get(call):
            self.fail[call] -= 1
            raise ColabError(f"colab {call} failed: no answer in 360 s")

    def alive(self) -> bool:
        return self.up

    def new(self, gpu: str) -> None:
        self.calls.append(f"new {gpu}")
        self.up = True
        self._maybe_fail("new")

    def stop(self) -> None:
        self._maybe_fail("stop")
        self.calls.append("stop")

    def upload(self, local: Path, remote: str) -> None:
        self.calls.append(f"upload {Path(remote).name}")
        if remote.endswith((".jsonl", ".json")):
            self.files[remote.split("/crisisline/", 1)[1]] = local.read_bytes()

    def download(self, remote: str, local: Path) -> None:
        if "/snapshot/" not in remote:
            local.write_text("log\n")
            return
        data = self.files[remote.split("/snapshot/", 1)[1]]
        if self.corrupt_downloads:
            self.corrupt_downloads -= 1
            data = data[:-1]
        local.write_bytes(data)

    def shell(self, script: str, timeout_s: float) -> str:
        if "head -n" not in script:  # setup or launch
            self.scripts.append(script)
            if f"-s {colab.TMUX_JOB}" not in script:
                self._maybe_fail("setup")
                self.calls.append("setup")
                return ""
            if self.lost_launches:
                self.lost_launches -= 1
                raise ColabError("colab exec failed: no answer in 360 s")
            self.calls.append("launch")
            self.running, self.polls_left = True, max(self.polls_left, 1)
            self._maybe_fail("launch")
            return ""
        if self.running:
            self.files[OUT] = self.files.get(OUT, b"") + b'{"r": 1}\n' * self.per_poll
            self.files.setdefault(MANIFEST, b"{}\n")
            self.polls_left -= 1
            self.running = self.polls_left > 0
        lines = []
        for path, data in self.files.items():
            snapshot = colab.snapshot_of_bytes(data)
            lines.append(f"{FILE_MARKER} {path} {snapshot.sha256} {snapshot.records}")
        state = "running" if self.running else "exit 0" if self.polls_left <= 0 else "none"
        return "\n".join([*lines, f"{JOB_MARKER} {state}", "INFO 2 / 4 records"])


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    path = tmp_path / "crisisline.bundle"
    path.write_bytes(b"bundle")
    return path


def test_fresh_runtime_runs_to_the_end_and_stops_after_the_verified_download(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls == [
        f"new {cfg['gpu']}",
        "upload crisisline.bundle",
        "setup",
        "launch",
        "stop",
    ]
    assert (tmp_path / OUT).read_bytes() == runtime.files[OUT]
    assert (tmp_path / OUT).read_bytes().count(b"\n") == 4


def test_a_fresh_runtime_resumes_from_the_local_outputs(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    for path, data in ((OUT, b'{"r": 0}\n' * 3), (MANIFEST, b"{}\n")):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(data)
    runtime = FakeRuntime()
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert "upload m.jsonl" in runtime.calls and "upload m.manifest.json" in runtime.calls
    assert (tmp_path / OUT).read_bytes().startswith(b'{"r": 0}\n' * 3)
    assert (tmp_path / OUT).read_bytes().count(b"\n") == 7


def test_a_running_job_is_followed_not_restarted(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime({OUT: b'{"r": 0}\n'})
    runtime.running = True
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls == ["stop"]


def test_an_unverified_download_never_replaces_the_local_copy_or_stops_the_runtime(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.corrupt_downloads = 10**6
    with pytest.raises(ColabError, match="no verified download"):
        colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None)
    assert "stop" not in runtime.calls
    assert not (tmp_path / OUT).exists()
    assert not list(tmp_path.rglob("*.download"))


def test_a_bad_download_is_tried_again(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.corrupt_downloads = cfg["download_attempts"] - 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert (tmp_path / OUT).read_bytes() == runtime.files[OUT]


def test_fewer_records_on_the_runtime_keep_the_local_copy(
    tmp_path: Path, cfg: dict[str, Any], bundle: Path
) -> None:
    (tmp_path / OUT).parent.mkdir(parents=True)
    (tmp_path / OUT).write_bytes(b"1\n2\n3\n")
    runtime = FakeRuntime({OUT: b"1\n"})
    status = Status(False, None, {OUT: Snapshot(colab.snapshot_of_bytes(b"1\n").sha256, 1)}, "")
    with pytest.raises(ColabError, match="local copy is kept"):
        colab.sync(runtime, status, {"workdir": "/content", "download_attempts": 3}, tmp_path)
    assert (tmp_path / OUT).read_bytes() == b"1\n2\n3\n"


def test_a_lost_runtime_is_not_stopped(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()

    def lost(script: str, timeout_s: float) -> str:
        raise ColabError("no exit code from the runtime")

    runtime.new(cfg["gpu"])
    runtime.shell = lost  # type: ignore[method-assign]
    with pytest.raises(ColabError):
        colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None)
    assert "stop" not in runtime.calls


def test_a_failed_poll_is_tried_again(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    shell, hangs = runtime.shell, iter([False, False, True, True])

    def flaky(script: str, timeout_s: float) -> str:
        if "head -n" in script and next(hangs, False):
            raise ColabError("colab exec failed: no answer in 360 s")
        return shell(script, timeout_s)

    runtime.shell = flaky  # type: ignore[method-assign]
    assert cfg["attempts"] == 3
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert (tmp_path / OUT).read_bytes() == runtime.files[OUT]
    assert runtime.calls[-1] == "stop"


def test_a_launch_that_went_through_despite_a_hang_is_followed_not_repeated(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.fail["launch"] = 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls.count("launch") == 1
    assert runtime.calls[-1] == "stop"


def test_a_launch_that_did_not_go_through_is_repeated(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.lost_launches = 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls.count("launch") == 1
    assert runtime.calls.count("setup") == 2
    assert (tmp_path / OUT).read_bytes() == runtime.files[OUT]


def test_a_runtime_created_by_a_hung_new_is_used(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.fail["new"] = 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls.count(f"new {cfg['gpu']}") == 1


def test_stop_is_tried_again(tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path) -> None:
    runtime = FakeRuntime()
    runtime.fail["stop"] = cfg["attempts"] - 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls[-1] == "stop"


def test_a_step_that_fails_every_attempt_stops_the_run_and_leaves_the_runtime(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    runtime.fail["setup"] = cfg["attempts"]
    waits: list[float] = []
    with pytest.raises(ColabError, match="launch generate failed 3 times in a row"):
        colab.run(runtime, job, cfg, tmp_path, bundle, sleep=waits.append)
    assert "stop" not in runtime.calls
    assert waits == [cfg["retry_wait_s"]] * (cfg["attempts"] - 1)


def test_a_repeated_launch_checks_outputs_uploaded_by_a_failed_one(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    (tmp_path / OUT).parent.mkdir(parents=True)
    (tmp_path / OUT).write_bytes(b'{"r": 0}\n')
    runtime = FakeRuntime()
    runtime.lost_launches = 1
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls.count("upload m.jsonl") == 1
    sha = colab.snapshot_of_bytes(b'{"r": 0}\n').sha256
    launches = [s for s in runtime.scripts if f"-s {colab.TMUX_JOB}" in s]
    assert len(launches) == 2 and all(f"{sha}  {OUT}" in s for s in launches)


def test_a_hung_cli_call_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("colab", 1)

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(ColabError, match="no answer in 1 s"):
        colab.Colab("s", 1).stop()


def cli_runner(stdout: str, returncode: int = 0) -> colab.Runner:
    def run(
        args: Sequence[str], stdin: str | None, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["colab", *args], returncode, stdout, "")

    return run


def test_shell_reads_the_exit_code_from_the_output() -> None:
    assert (
        colab.Colab("s", 60, cli_runner(f"hello\n{colab.EXIT_MARKER}=0\n")).shell("x", 1)
        == "hello\n"
    )
    with pytest.raises(ColabError, match="exit 3"):
        colab.Colab("s", 60, cli_runner(f"{colab.EXIT_MARKER}=3\n")).shell("x", 1)
    with pytest.raises(ColabError, match="no exit code"):
        colab.Colab("s", 60, cli_runner("Traceback ...\n")).shell("x", 1)
    with pytest.raises(ColabError, match="colab exec failed"):
        colab.Colab("s", 60, cli_runner("lost", returncode=1)).shell("x", 1)


def test_cli_calls_name_the_session() -> None:
    calls: list[list[str]] = []

    def run(
        args: Sequence[str], stdin: str | None, timeout_s: float
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(["colab", *args], 0, "", "")

    runtime = colab.Colab("s", 60, run)
    runtime.new("L4")
    runtime.upload(Path("crisisline.bundle"), "/content/crisisline.bundle")
    runtime.download("/content/generate.log", Path("generate.log"))
    runtime.stop()
    assert calls == [
        ["new", "-s", "s", "--gpu", "L4"],
        ["upload", "-s", "s", "crisisline.bundle", "/content/crisisline.bundle"],
        ["download", "-s", "s", "/content/generate.log", "generate.log"],
        ["stop", "-s", "s"],
    ]


def test_a_job_gone_without_an_exit_code_stops_the_run_and_leaves_the_runtime(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    shell = runtime.shell

    def vanishing(script: str, timeout_s: float) -> str:
        out = shell(script, timeout_s)
        runtime.running = False  # the tmux session died without writing generate.exit
        return out

    runtime.shell = vanishing  # type: ignore[method-assign]
    with pytest.raises(ColabError, match="neither running nor finished"):
        colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None)
    assert runtime.calls == [f"new {cfg['gpu']}", "upload crisisline.bundle", "setup", "launch"]


def test_alive_matches_the_session_name_only() -> None:
    listing = "[crisisline-m-main2] gpu-l4 | Hardware: L4\n[colab] note\n"
    assert not colab.Colab("crisisline-m-main", 60, cli_runner(listing)).alive()
    assert colab.Colab("crisisline-m-main2", 60, cli_runner(listing)).alive()


def test_parse_status() -> None:
    out = f"{FILE_MARKER} {OUT} {'b' * 64} 12\n{JOB_MARKER} exit 1\nlast log line\n"
    assert colab.parse_status(out) == Status(
        False, 1, {OUT: Snapshot("b" * 64, 12)}, "last log line"
    )
    assert colab.parse_status(f"{JOB_MARKER} running\n").running
    assert colab.parse_status("") == Status(False, None, {}, "")


def test_an_exit_code_not_written_yet_is_a_colab_error() -> None:
    """A poll between the creation of generate.exit and the write of the code into it: an
    error the poll step tries again, not a crash of the driver."""
    for line in (f"{JOB_MARKER} exit", f"{JOB_MARKER} exit x"):
        with pytest.raises(ColabError, match="no exit code"):
            colab.parse_status(line + "\n")


def test_an_empty_exit_file_is_polled_again(
    tmp_path: Path, cfg: dict[str, Any], job: Job, bundle: Path
) -> None:
    runtime = FakeRuntime()
    shell, empty = runtime.shell, iter([False, False, True])

    def writing_exit_code(script: str, timeout_s: float) -> str:
        out = shell(script, timeout_s)
        if "head -n" in script and next(empty, False):
            return out.replace(f"{JOB_MARKER} exit 0", f"{JOB_MARKER} exit")
        return out

    runtime.shell = writing_exit_code  # type: ignore[method-assign]
    assert colab.run(runtime, job, cfg, tmp_path, bundle, sleep=lambda _: None) == 0
    assert runtime.calls[-1] == "stop"


def test_scripts_are_valid_bash(cfg: dict[str, Any], job: Job) -> None:
    for script in (
        colab.setup_script(job, cfg),
        colab.launch_script(job, cfg, {OUT: "c" * 64}),
        colab.status_script(job, cfg),
    ):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_a_job_that_ends_during_the_snapshot_is_not_reported_finished(
    tmp_path: Path, cfg: dict[str, Any], job: Job
) -> None:
    """generate writes its last record and exits while the status script counts the lines: the
    poll must not report the end with a snapshot that lacks that record (the runtime would be
    stopped and the record lost)."""
    work = tmp_path / "content"
    out = work / "crisisline" / OUT
    out.parent.mkdir(parents=True)
    out.write_bytes(b'{"r": 1}\n')
    shims = tmp_path / "bin"
    shims.mkdir()
    wc = shims / "wc"
    wc.write_text(
        "#!/bin/bash\n"
        f'/usr/bin/wc "$@"\n'
        f"if [ ! -f {tmp_path}/ended ]; then touch {tmp_path}/ended;"
        f" printf '{{\"r\": 2}}\\n' >> {out}; echo 0 > {work}/generate.exit; fi\n"
    )
    wc.chmod(0o755)
    script = colab.status_script(job, cfg | {"workdir": str(work)})
    env = {"PATH": f"{shims}:/usr/bin:/bin:/usr/sbin:/sbin"}
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=True
    )
    status = colab.parse_status(result.stdout)
    assert status.exit_code is None or status.files[OUT].records == 2


def test_setup_checks_the_gpu_and_the_commit(cfg: dict[str, Any], job: Job) -> None:
    script = colab.setup_script(job, cfg)
    assert f"grep -q {cfg['gpu']}" in script
    assert f"checkout -q {job.commit}" in script
    assert "ollama pull qwen3.5:4b-q8_0" in script
