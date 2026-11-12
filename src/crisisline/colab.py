"""Run ``crisisline generate`` for one local model on a Google Colab runtime.

Driven from this machine with the ``colab`` CLI (google-colab-cli). The exact commit reaches the
runtime as a git bundle, so no token or key ever goes to Colab. On the runtime the repository
is cloned, uv and Ollama are installed, the model is pulled, and ``generate`` runs in a detached
tmux session that depends neither on the kernel nor on this machine.

The local copy of the outputs is the one that counts. Every ``poll_s`` the runtime snapshots
the complete lines of each output, and the snapshot replaces the local file only once its
download has the sha256 and record count the runtime reported. A fresh runtime gets the local
outputs uploaded (and checked) before ``generate`` resumes, so a lost runtime costs at most the
records since the last download. The runtime is stopped only after the final download is
verified.

Every step (start the runtime, set up and launch generate, poll, stop) is safe to repeat and is
tried up to ``attempts`` times in a row before the run gives up: launching starts with a fresh
poll, so a launch that went through despite an error is followed, not repeated.

``colab exec`` exits 0 even when the code it ran raised, so every command on the runtime prints
its exit code after :data:`EXIT_MARKER`, and :meth:`Colab.shell` reads it from there.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

EXIT_MARKER = "__crisisline_exit__"
FILE_MARKER = "__crisisline_file__"
JOB_MARKER = "__crisisline_job__"
TMUX_JOB = "generate"
TMUX_OLLAMA = "ollama"


class ColabError(RuntimeError):
    """A colab CLI call failed, the runtime is gone, or a command on it failed."""


@dataclass(frozen=True)
class Job:
    """One ``generate`` run of a local model on a runtime."""

    session: str
    commit: str
    tag: str  # Ollama tag
    ollama_version: str  # "" installs the latest
    generate_args: tuple[str, ...]  # arguments of `crisisline generate`
    outputs: tuple[str, ...]  # repository-relative: generations file and its manifest


@dataclass(frozen=True)
class Snapshot:
    sha256: str
    records: int


@dataclass(frozen=True)
class Status:
    running: bool
    exit_code: int | None  # of the last finished generate
    files: dict[str, Snapshot]  # outputs present on the runtime
    log_tail: str


Runner = Callable[[Sequence[str], str | None, float], subprocess.CompletedProcess[str]]


def run_cli(
    args: Sequence[str], stdin: str | None, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    """The colab CLI; a call that hangs (``colab exec`` can) fails after ``timeout_s``."""
    command = ["colab", *args]
    try:
        return subprocess.run(
            command, input=stdin, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(command, 124, "", f"no answer in {timeout_s} s")


class Colab:
    """The colab CLI, bound to one named session."""

    def __init__(self, session: str, timeout_s: float, runner: Runner = run_cli) -> None:
        self.session = session
        self.timeout_s = timeout_s
        self._run = runner

    def _call(self, *args: str, stdin: str | None = None, timeout_s: float | None = None) -> str:
        result = self._run(args, stdin, timeout_s or self.timeout_s)
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            raise ColabError(f"colab {args[0]} failed: {output[-1000:]}")
        return result.stdout

    def alive(self) -> bool:
        listing = self._call("sessions")
        return any(line.startswith(f"[{self.session}] ") for line in listing.splitlines())

    def new(self, gpu: str) -> None:
        self._call("new", "-s", self.session, "--gpu", gpu)

    def stop(self) -> None:
        self._call("stop", "-s", self.session)

    def upload(self, local: Path, remote: str) -> None:
        self._call("upload", "-s", self.session, str(local), remote)

    def download(self, remote: str, local: Path) -> None:
        self._call("download", "-s", self.session, remote, str(local))

    def shell(self, script: str, timeout_s: float) -> str:
        """Run a bash script on the runtime and return its output; it must exit 0."""
        code = (
            "import subprocess\n"
            f"_r = subprocess.run(['bash', '-c', {script!r}], stdout=subprocess.PIPE,"
            " stderr=subprocess.STDOUT, text=True)\n"
            "print(_r.stdout, end='')\n"
            f"print('\\n{EXIT_MARKER}=' + str(_r.returncode))\n"
        )
        out = self._call(
            "exec",
            "-s",
            self.session,
            "--timeout",
            str(timeout_s),
            stdin=code,
            timeout_s=timeout_s + 60,
        )
        match = re.search(rf"^{EXIT_MARKER}=(\d+)$", out, re.MULTILINE)
        if match is None:
            raise ColabError(f"no exit code from the runtime (lost or timed out): {out[-1000:]}")
        if match[1] != "0":
            raise ColabError(f"command failed on the runtime (exit {match[1]}): {out[-2000:]}")
        return out[: match.start()]


# --- Scripts run on the runtime ---------------------------------------------------------------


def _repo(cfg: Mapping[str, Any]) -> str:
    return f"{cfg['workdir']}/crisisline"


def setup_script(job: Job, cfg: Mapping[str, Any]) -> str:
    """Clone the bundled commit, install uv and Ollama, serve and pull the model, build prompts.

    Idempotent on a live runtime, which must hold the same commit.
    """
    q = shlex.quote
    work, repo, gpu = cfg["workdir"], _repo(cfg), cfg["gpu"]
    dirs = sorted({str(Path(path).parent) for path in job.outputs})
    return "\n".join(
        [
            "set -euo pipefail",
            f"nvidia-smi --query-gpu=name --format=csv,noheader | grep -q {q(gpu)}"
            f" || {{ echo 'not a {gpu} runtime:'; nvidia-smi -L; exit 1; }}",
            f"if [ ! -d {repo} ]; then git clone -q {work}/crisisline.bundle {repo};"
            f" git -C {repo} checkout -q {job.commit}; fi",
            f'[ "$(git -C {repo} rev-parse HEAD)" = {job.commit} ]'
            " || { echo 'the runtime holds another commit; stop it first'; exit 1; }",
            'export PATH="$HOME/.local/bin:$PATH"',
            f"command -v uv > /dev/null || curl -LsSf {q(cfg['install']['uv'])} | sh",
            f"cd {repo}",
            "uv sync --locked --quiet",
            # The Ollama installer unpacks a .tar.zst.
            "command -v zstd > /dev/null || { apt-get -qq update && apt-get -qq install -y zstd; }",
            f"command -v ollama > /dev/null || curl -fsSL {q(cfg['install']['ollama'])}"
            f" | OLLAMA_VERSION={q(job.ollama_version)} sh",
            f"tmux has-session -t {TMUX_OLLAMA} 2> /dev/null"
            f" || tmux new-session -d -s {TMUX_OLLAMA} 'ollama serve >> {work}/ollama.log 2>&1'",
            "for _ in $(seq 60); do ollama list > /dev/null 2>&1 && break; sleep 1; done",
            "ollama --version",
            f"ollama pull {q(job.tag)} 2>&1 | tail -n 1",
            "uv run crisisline build",
            f"mkdir -p {' '.join(q(d) for d in dirs)}",
        ]
    )


def launch_script(job: Job, cfg: Mapping[str, Any], outputs: Mapping[str, str]) -> str:
    """Check the runtime's outputs against these sha256, then start generate in a detached tmux
    session.

    The job writes generate's exit code to ``generate.exit`` when it ends.
    """
    q = shlex.quote
    work, repo = cfg["workdir"], _repo(cfg)
    generate = shlex.join(["uv", "run", "crisisline", "generate", *job.generate_args])
    job_sh = (
        f'export PATH="$HOME/.local/bin:$PATH"\ncd {repo}\n'
        f"{generate} >> {work}/generate.log 2>&1\necho $? > {work}/generate.exit\n"
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo}",
            *(
                f"echo {q(f'{sha}  {path}')} | sha256sum --check --quiet"
                for path, sha in outputs.items()
            ),
            f"rm -f {work}/generate.exit",
            f"cat > {work}/generate.sh << 'EOF'\n{job_sh}EOF",
            f"tmux new-session -d -s {TMUX_JOB} 'bash {work}/generate.sh'",
        ]
    )


def status_script(job: Job, cfg: Mapping[str, Any]) -> str:
    """Snapshot the complete lines of each output and report the job.

    Prints one :data:`FILE_MARKER` line per output on the runtime, one :data:`JOB_MARKER` line,
    then the end of generate's log. Prints nothing on a runtime without the repository.

    The job is looked at before the snapshot: a job that has ended by then has written all its
    records, while one that ends during the snapshot is still reported running.
    """
    q = shlex.quote
    work, repo = cfg["workdir"], _repo(cfg)
    exit_file = f"{work}/generate.exit"
    lines = [
        "set -euo pipefail",
        f"cd {repo} 2> /dev/null || exit 0",
        f'if [ -f {exit_file} ]; then job="exit $(cat {exit_file})";'
        f" elif tmux has-session -t {TMUX_JOB} 2> /dev/null; then job=running;"
        " else job=none; fi",
    ]
    for path in job.outputs:
        snap = q(f"{work}/snapshot/{path}")
        lines.append(
            f'if [ -f {q(path)} ]; then mkdir -p "$(dirname {snap})";'
            f' head -n "$(wc -l < {q(path)})" {q(path)} > {snap};'
            f' echo "{FILE_MARKER} {path} $(sha256sum < {snap} | cut -c1-64) $(wc -l < {snap})"; fi'
        )
    lines += [
        f'echo "{JOB_MARKER} $job"',
        f"tail -n 2 {work}/generate.log 2> /dev/null || true",
    ]
    return "\n".join(lines)


def parse_status(out: str) -> Status:
    files: dict[str, Snapshot] = {}
    running, exit_code, tail = False, None, []
    for line in out.splitlines():
        fields = line.split()
        if fields[:1] == [FILE_MARKER]:
            files[fields[1]] = Snapshot(fields[2], int(fields[3]))
        elif fields[:1] == [JOB_MARKER]:
            running = fields[1] == "running"
            if fields[1] == "exit" and not (len(fields) > 2 and fields[2].isdigit()):
                # generate.exit exists but its code is not written yet
                raise ColabError(f"no exit code in generate.exit yet: {line!r}")
            exit_code = int(fields[2]) if fields[1] == "exit" else None
        elif line.strip():
            tail.append(line)
    return Status(running, exit_code, files, "\n".join(tail))


# --- Local side -------------------------------------------------------------------------------


def snapshot_of_bytes(data: bytes) -> Snapshot:
    return Snapshot(hashlib.sha256(data).hexdigest(), data.count(b"\n"))


def snapshot_of(path: Path) -> Snapshot:
    return snapshot_of_bytes(path.read_bytes())


def sync(colab: Colab, status: Status, cfg: Mapping[str, Any], root: Path) -> None:
    """Replace each local output with the runtime's snapshot once its download is verified.

    Raises :class:`ColabError` if the runtime has fewer records than the local copy, or if no
    download in ``download_attempts`` matches the snapshot's sha256 and record count.
    """
    for path, remote in status.files.items():
        local = root / path
        before = snapshot_of(local) if local.exists() else Snapshot("", 0)
        if before == remote:
            continue
        if remote.records < before.records:
            raise ColabError(
                f"{path}: {remote.records} records on the runtime, {before.records} locally;"
                " the local copy is kept"
            )
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".download")
        for attempt in range(1, cfg["download_attempts"] + 1):
            colab.download(f"{cfg['workdir']}/snapshot/{path}", tmp)
            got = snapshot_of(tmp)
            if got == remote:
                break
            logger.warning(
                "%s: download %d does not match the runtime: %s != %s", path, attempt, got, remote
            )
        else:
            tmp.unlink(missing_ok=True)
            raise ColabError(
                f"{path}: no verified download after {cfg['download_attempts']} attempts"
            )
        tmp.replace(local)
        logger.info("%s: %d records, sha256 %s, verified", path, remote.records, remote.sha256[:12])


def poll(colab: Colab, job: Job, cfg: Mapping[str, Any], root: Path, log: Path) -> Status:
    """Snapshot the runtime's outputs, download and verify them, and fetch generate's log."""
    status = parse_status(colab.shell(status_script(job, cfg), cfg["exec_timeout_s"]))
    sync(colab, status, cfg, root)
    if status.log_tail:
        logger.info("runtime: %s", status.log_tail.replace("\n", " | "))
        log.parent.mkdir(parents=True, exist_ok=True)
        colab.download(f"{cfg['workdir']}/generate.log", log)
    return status


def start(
    colab: Colab, job: Job, cfg: Mapping[str, Any], root: Path, bundle: Path, on_runtime: Status
) -> None:
    """Set the runtime up, upload the local outputs it lacks, and launch generate.

    The launch first checks every local output's sha256 against the runtime's copy, so a
    repeated start checks outputs uploaded by an earlier, failed one too.
    """
    colab.upload(bundle, f"{cfg['workdir']}/crisisline.bundle")
    logger.info(
        "Setting up %s: commit %s, %s (a model pull takes minutes)",
        job.session,
        job.commit[:12],
        job.tag,
    )
    logger.info("%s", colab.shell(setup_script(job, cfg), cfg["setup_timeout_s"]).strip())
    local_outputs: dict[str, str] = {}
    for path in job.outputs:
        local = root / path
        if not local.exists():
            continue
        if path not in on_runtime.files:
            colab.upload(local, f"{_repo(cfg)}/{path}")
            logger.info("Uploaded the local %s to resume from it", path)
        local_outputs[path] = snapshot_of(local).sha256
    colab.shell(launch_script(job, cfg, local_outputs), cfg["exec_timeout_s"])
    logger.info("generate %s started in tmux session '%s'", shlex.join(job.generate_args), TMUX_JOB)


T = TypeVar("T")


def attempt(
    step: str, action: Callable[[], T], cfg: Mapping[str, Any], sleep: Callable[[float], None]
) -> T:
    """``action()``, tried up to ``attempts`` times in a row while it raises :class:`ColabError`."""
    attempts = cfg["attempts"]
    for n in range(1, attempts + 1):
        try:
            return action()
        except ColabError as exc:
            if n == attempts:
                raise ColabError(f"{step} failed {attempts} times in a row: {exc}") from exc
            logger.warning(
                "%s failed (%d of %d), trying again in %d s: %s",
                step,
                n,
                attempts,
                cfg["retry_wait_s"],
                exc,
            )
            sleep(cfg["retry_wait_s"])
    raise AssertionError("attempts must be at least 1")


def run(
    colab: Colab,
    job: Job,
    cfg: Mapping[str, Any],
    root: Path,
    bundle: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run generate on the job's runtime until it ends; return generate's exit code.

    Reuses a live runtime and follows a generate already running there; otherwise starts one,
    which resumes from the local outputs. The runtime is stopped after the final download is
    verified. On :class:`ColabError` (a step failed ``attempts`` times in a row, or generate is
    neither running nor finished) it is left as is; rerunning the same command resumes.
    """
    started = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    log = root / cfg["logs"] / f"{job.session}.{started}.log"

    def ensure_runtime() -> None:
        if colab.alive():
            logger.info("Runtime %s is up; using it", job.session)
        else:
            logger.info("Starting runtime %s (%s)", job.session, cfg["gpu"])
            colab.new(cfg["gpu"])

    def ensure_generate() -> None:
        status = poll(colab, job, cfg, root, log)
        if status.running:
            logger.info("generate is running on %s; following it", job.session)
        else:
            start(colab, job, cfg, root, bundle, status)

    def stop() -> None:
        if colab.alive():
            colab.stop()

    attempt("start runtime", ensure_runtime, cfg, sleep)
    attempt("launch generate", ensure_generate, cfg, sleep)
    while True:
        status = attempt("poll", lambda: poll(colab, job, cfg, root, log), cfg, sleep)
        if status.exit_code is not None:
            break
        if not status.running:
            raise ColabError("generate is neither running nor finished on the runtime")
        sleep(cfg["poll_s"])
    logger.info(
        "generate exited with %d; outputs verified, stopping %s", status.exit_code, job.session
    )
    attempt("stop runtime", stop, cfg, sleep)
    return status.exit_code
