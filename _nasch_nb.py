"""Import shim for notebooks/nagel_schreckenberg.ipynb.

Re-exports the single-lane NaSch classes, constants, and helpers so other .py
files can `from _nasch_nb import TrafficCA, TrafficCA_Periodic, ...` without
copy-pasting the notebook code. Only class/function definitions, imports,
and module-level scalar assignments are executed; plotting, sweeps, and
print statements at notebook cell level are filtered out so the shim loads
quickly and silently.
"""

import ast
import warnings
from pathlib import Path

import nbformat

_NB_PATH = Path(__file__).resolve().parent / "notebooks" / "nagel_schreckenberg.ipynb"


def _load_notebook_namespace():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = nbformat.read(str(_NB_PATH), as_version=4)

    keep_types = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Assign,
        ast.AnnAssign,
    )
    ns = {"__name__": "_nasch_nb_loaded"}

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue

        kept = []
        for node in tree.body:
            if not isinstance(node, keep_types):
                continue
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Tuple) for t in node.targets):
                    continue
                # Only keep module-level assignments whose RHS is a literal.
                # This preserves constants (`L = 500`, `P_RAND = 0.3`) while
                # dropping analysis side-effects (`sim = TrafficCAWithLights(...)`,
                # `grid = record_spacetime(...)`, `fig = plt.figure(...)`).
                try:
                    ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
            kept.append(node)

        if not kept:
            continue

        module = ast.Module(body=kept, type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, filename=f"<nagel_schreckenberg.ipynb:cell{idx}>", mode="exec")
        exec(code, ns)

    return ns


_ns = _load_notebook_namespace()

TrafficCA = _ns["TrafficCA"]
TrafficCAWithLights = _ns["TrafficCAWithLights"]
TrafficCA_Periodic = _ns["TrafficCA_Periodic"]
fundamental_diagram = _ns["fundamental_diagram"]
sweep_inflow = _ns["sweep_inflow"]
record_spacetime = _ns["record_spacetime"]

L = _ns["L"]
V_MAX = _ns["V_MAX"]
P_RAND = _ns["P_RAND"]
T_WARMUP = _ns["T_WARMUP"]
T_MEASURE = _ns["T_MEASURE"]

__all__ = [
    "TrafficCA",
    "TrafficCAWithLights",
    "TrafficCA_Periodic",
    "fundamental_diagram",
    "sweep_inflow",
    "record_spacetime",
    "L",
    "V_MAX",
    "P_RAND",
    "T_WARMUP",
    "T_MEASURE",
]
