"""The paper's figures, tables and cited numbers, built from the analysis outputs."""

import os

# The figures are only written to files, with Agg. Colab and Jupyter set MPLBACKEND to their
# inline backend, which is not installed here, so importing matplotlib would fail; this runs before
# any module of the package imports it.
os.environ["MPLBACKEND"] = "agg"
