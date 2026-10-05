"""
Restricted execution of agent-written pandas code.

Three layers, each catching what the one before misses:
  1. AST validation — no imports, no file/network/process access, no dunder
     attribute access (the usual route out of a restricted namespace).
  2. Restricted builtins — the code runs with a whitelist, not __builtins__.
  3. A separate process with a hard timeout and a copy of the data, so a
     runaway loop or a crash cannot take the investigation down with it.

This is defense in depth for a trusted-user tool, NOT a security boundary for
untrusted input. A multi-tenant deployment should run this in a real sandbox
(gVisor / Firecracker container) or use Anthropic's server-side code
execution tool instead.
"""

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile

import pandas as pd

TIMEOUT_S = 20
MAX_OUTPUT_CHARS = 4000

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "classmethod", "staticmethod", "super", "type", "object",
}
BLOCKED_ATTR_PREFIXES = ("read_", "to_csv", "to_pickle", "to_parquet", "to_excel", "to_sql",
                         "to_hdf", "to_feather", "to_stata", "to_clipboard", "to_json", "tofile")
BLOCKED_ATTRS = {"load", "save", "savez", "fromfile", "system", "popen", "eval", "query"}

SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "int",
    "isinstance", "len", "list", "map", "max", "min", "print", "range", "repr", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "KeyError", "Exception",
]


class UnsafeCode(ValueError):
    pass


def validate(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCode(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("imports are not allowed; pd, np and stats (scipy.stats) are preloaded")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCode(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise UnsafeCode(f"'{node.id}' is not allowed")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                raise UnsafeCode(f"access to private attribute '{attr}' is not allowed")
            if attr in BLOCKED_ATTRS or attr.startswith(BLOCKED_ATTR_PREFIXES):
                raise UnsafeCode(f"'.{attr}' (file or process access) is not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise UnsafeCode("string constants containing '__' are not allowed")


# Runs in the child process. The last expression's value is printed, notebook-style.
_RUNNER = r"""
import ast, builtins, json, pickle, sys, io, contextlib
import numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
payload = json.loads(sys.stdin.read())
with open(payload["data"], "rb") as f:
    df = pickle.load(f)
safe = {name: getattr(builtins, name) for name in payload["builtins"] if hasattr(builtins, name)}
env = {"__builtins__": safe, "pd": pd, "np": np, "stats": stats, "df": df}
tree = ast.parse(payload["code"])
last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(tree, "<agent>", "exec"), env)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<agent>", "eval"), env)
        if value is not None:
            print(value)
sys.stdout.write(buf.getvalue())
"""


def run_python(code: str, df: pd.DataFrame, timeout: int = TIMEOUT_S) -> dict:
    validate(code)
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "df.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(df, f)
        payload = json.dumps({"code": code, "data": data_path, "builtins": SAFE_BUILTINS})
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER], input=payload, capture_output=True,
                text=True, timeout=timeout, cwd=tmp,
                env={k: v for k, v in os.environ.items()
                     if k in ("PATH", "SYSTEMROOT", "PYTHONPATH", "TEMP", "TMP")},
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": f"timed out after {timeout}s"}

    stdout = proc.stdout
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(proc.stdout)} chars total]"
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return {"ok": False, "stdout": stdout, "error": err[-1] if err else "process failed"}
    return {"ok": True, "stdout": stdout, "error": None}
