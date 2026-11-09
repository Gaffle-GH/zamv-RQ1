"""Build the RQ1 dataset: 300 bug-injected functions + 100 correct controls (HumanEval + MBPP-sanitized).

Each bug is one AST mutation that fails at least one unit test:
  off_by_one          an integer literal +-1, or a non-literal range()/slice bound +-1
  wrong_boundary      comparison strictness swapped (< <-> <=, > <-> >=)
  flipped_comparison  comparison direction or negation flipped (< <-> >, == <-> !=, in <-> not in, ...)
  removed_check       an if-block without else deleted, or one operand of an and/or dropped
Mutants where some tests still pass are preferred (subtle bugs). Buggy and correct code are both
re-printed with ast.unparse, so formatting never leaks which item is buggy.

    python -m rq1.dataset            # -> data/items.jsonl, data/items.csv
"""

import argparse
import ast
import copy
import csv
import difflib
import gzip
import json
import random
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .common import DATA, write_jsonl

SOURCES = {
    "HumanEval.jsonl.gz": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    "sanitized-mbpp.json": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
}
warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escapes in MBPP sources
BUG_TYPES = ["off_by_one", "wrong_boundary", "flipped_comparison", "removed_check"]

# Runs `code` then `tests` with every assert turned into a recorded check, prints one JSON line.
RUNNER = r'''
import ast, json, sys
code, tests = json.load(sys.stdin)
res = []
def __rec(f):
    try: res.append(bool(f()))
    except Exception: res.append(False)
class T(ast.NodeTransformer):
    def visit_Assert(self, n):
        lam = ast.Lambda(ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), n.test)
        return ast.copy_location(ast.Expr(ast.Call(ast.Name("__rec", ast.Load()), [lam], [])), n)
g = {"__rec": __rec, "__name__": "__main__"}
err = None
try:
    exec(code, g)
    exec(compile(ast.fix_missing_locations(T().visit(ast.parse(tests))), "<tests>", "exec"), g)
except BaseException as e:
    err = type(e).__name__
print("\n" + json.dumps({"n": len(res), "passed": sum(res), "error": err}))
'''


def run_tests(code, tests, timeout=10):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps([code, tests]),
                               capture_output=True, text=True, timeout=timeout, cwd=tmp)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return {"n": 0, "passed": 0, "error": "Timeout"}
        except (json.JSONDecodeError, IndexError):
            return {"n": 0, "passed": 0, "error": "Crash"}


def passes(r):
    return r["error"] is None and r["n"] > 0 and r["passed"] == r["n"]


# ----------------------------------------------------------------- loading


def fetch():
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (DATA / "raw" / name).exists():
            urllib.request.urlretrieve(url, DATA / "raw" / name)


def strip_docstring(fn):
    doc = ast.get_docstring(fn, clean=True)
    if doc is not None:
        fn.body = fn.body[1:] or [ast.Pass()]
    return doc


def load_problems():
    """Uniform records: item_id, source, task_id, entry_point, spec, code (docstring-free), tests."""
    out = []
    with gzip.open(DATA / "raw" / "HumanEval.jsonl.gz", "rt") as f:
        for line in f:
            p = json.loads(line)
            tree = ast.parse(p["prompt"] + p["canonical_solution"])
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == p["entry_point"])
            spec = strip_docstring(fn)
            out.append(dict(item_id=f"humaneval-{p['task_id'].split('/')[1]}", source="humaneval",
                            task_id=p["task_id"], entry_point=p["entry_point"], spec=spec,
                            code=ast.unparse(tree), tests=p["test"] + f"\n\ncheck({p['entry_point']})\n"))
    for p in json.load(open(DATA / "raw" / "sanitized-mbpp.json")):
        tree = ast.parse(p["code"])
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        called = [n.func.id for n in ast.walk(ast.parse(p["test_list"][0]))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined]
        if not called:
            continue
        entry = called[0]
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry)
        strip_docstring(fn)
        spec = f"{p['prompt'].strip()}\n\nExample test:\n{p['test_list'][0]}"
        out.append(dict(item_id=f"mbpp-{p['task_id']}", source="mbpp", task_id=str(p["task_id"]),
                        entry_point=entry, spec=spec, code=ast.unparse(tree),
                        tests="\n".join(p["test_imports"] + p["test_list"]) + "\n"))
    return out


# ---------------------------------------------------------------- mutation

CMP = {  # op -> [(replacement, bug_type)]
    ast.Lt: [(ast.LtE, "wrong_boundary"), (ast.Gt, "flipped_comparison")],
    ast.LtE: [(ast.Lt, "wrong_boundary"), (ast.GtE, "flipped_comparison")],
    ast.Gt: [(ast.GtE, "wrong_boundary"), (ast.Lt, "flipped_comparison")],
    ast.GtE: [(ast.Gt, "wrong_boundary"), (ast.LtE, "flipped_comparison")],
    ast.Eq: [(ast.NotEq, "flipped_comparison")],
    ast.NotEq: [(ast.Eq, "flipped_comparison")],
    ast.In: [(ast.NotIn, "flipped_comparison")],
    ast.NotIn: [(ast.In, "flipped_comparison")],
    ast.Is: [(ast.IsNot, "flipped_comparison")],
    ast.IsNot: [(ast.Is, "flipped_comparison")],
}


def function_nodes(tree):
    """Nodes inside function bodies, in ast.walk order (stable across deep copies)."""
    inside = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for stmt in n.body:
                inside.update(id(m) for m in ast.walk(stmt))
    return [(i, n) for i, n in enumerate(ast.walk(tree)) if id(n) in inside]


def sites(tree):
    """Every possible mutation as (node_index, bug_type, op)."""
    out = []
    for i, n in function_nodes(tree):
        if isinstance(n, ast.Compare):
            for j, op in enumerate(n.ops):
                out += [(i, kind, ("cmp", j, new)) for new, kind in CMP.get(type(op), [])]
        elif isinstance(n, ast.Constant) and type(n.value) is int:
            out += [(i, "off_by_one", ("const", d)) for d in (1, -1)]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" and n.args:
            stop = 0 if len(n.args) == 1 else 1
            if not isinstance(n.args[stop], ast.Constant):
                out += [(i, "off_by_one", ("range", stop, d)) for d in (1, -1)]
        elif isinstance(n, ast.Slice):
            for part in ("lower", "upper"):
                b = getattr(n, part)
                if b is not None and not isinstance(b, (ast.Constant, ast.UnaryOp)):
                    out += [(i, "off_by_one", ("slice", part, d)) for d in (1, -1)]
        elif isinstance(n, ast.If) and not n.orelse:
            out.append((i, "removed_check", ("del_if",)))
        elif isinstance(n, ast.BoolOp):
            out += [(i, "removed_check", ("drop", k)) for k in range(len(n.values))]
    return out


def plus(expr, d):
    return ast.BinOp(expr, ast.Add() if d > 0 else ast.Sub(), ast.Constant(1))


def apply(code, site):
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    i, _, op = site
    n = nodes[i]
    if op[0] == "cmp":
        n.ops[op[1]] = op[2]()
    elif op[0] == "const":
        n.value += op[1]
    elif op[0] == "range":
        n.args[op[1]] = plus(n.args[op[1]], op[2])
    elif op[0] == "slice":
        setattr(n, op[1], plus(getattr(n, op[1]), op[2]))
    elif op[0] == "drop":
        n.values = n.values[: op[1]] + n.values[op[1] + 1:]  # a 1-value BoolOp unparses as that value
    elif op[0] == "del_if":
        for parent in nodes:
            for field in ("body", "orelse", "finalbody"):
                block = getattr(parent, field, None)
                if isinstance(block, list) and any(s is n for s in block):
                    if len(block) == 1:
                        raise ValueError("would leave a bare `pass` block, an obvious bug")
                    block[:] = [s for s in block if s is not n]
    return ast.unparse(ast.fix_missing_locations(tree))


def changed_line(ref, mut):
    """1-based line in `mut` where it first differs from `ref`, plus the old and new text."""
    a, b = ref.splitlines(), mut.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            j = min(j1, len(b) - 1)
            return j + 1, "\n".join(a[i1:i2]).strip(), b[j].strip()
    return None


def mutants(problem, rng, max_sites=24):
    """All distinct test-failing mutants of one problem (a sample of sites, for speed)."""
    s = sites(ast.parse(problem["code"]))
    rng.shuffle(s)
    seen, out = {problem["code"]}, []
    for site in s:
        try:
            code = apply(problem["code"], site)
            compile(code, "<mutant>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append(dict(code=code, bug_type=site[1], op=str(site[2][0])))
        if len(out) >= max_sites:
            break
    return out


# -------------------------------------------------------------------- build


def build(n_buggy=300, n_correct=100, seed=0, workers=8):
    rng = random.Random(seed)
    fetch()
    problems = load_problems()
    print(f"{len(problems)} problems loaded")
    with ThreadPoolExecutor(workers) as ex:
        ref = list(ex.map(lambda p: run_tests(p["code"], p["tests"]), problems))
        for p, r in zip(problems, ref):
            p["n_tests"] = r["n"]  # count asserts on the reference: a crashing mutant can record 0
        problems = [p for p, r in zip(problems, ref) if passes(r)]
        print(f"{len(problems)} reference solutions pass their tests")
        cands = [(p, m) for p in problems for m in mutants(p, rng)]
        results = list(ex.map(lambda pm: run_tests(pm[1]["code"], pm[0]["tests"]), cands))
    by_item = {}
    for (p, m), r in zip(cands, results):
        if r["error"] == "Timeout" or passes(r):
            continue  # equivalent mutant, or an infinite loop the verifier cannot reason about
        m.update(n_tests=r["n"], n_passed=r["passed"], test_error=r["error"],
                 subtle=r["error"] is None and r["passed"] > 0)
        by_item.setdefault(p["item_id"], []).append(m)

    pool = [p for p in problems if p["item_id"] in by_item]
    rng.shuffle(pool)
    if len(pool) < n_buggy + n_correct:
        raise SystemExit(f"only {len(pool)} usable problems")
    items = []
    for p in pool[:n_correct]:  # controls are a random draw, chosen before bug balancing
        # Controls get a focus line (where a bug could have gone) so A4/FALSE_ALARM target a real line.
        line, _, _ = changed_line(p["code"], rng.choice(by_item[p["item_id"]])["code"])
        ref_lines = p["code"].splitlines()
        items.append(dict(p, reference_code=p["code"], is_buggy=False, bug_type=None, mutation=None,
                          original_line_text=None, focus_line=line,
                          focus_line_text=ref_lines[min(line, len(ref_lines)) - 1].strip(),
                          n_tests_failed=0, test_error=None))
    # Buggy items: equal quota per bug type, subtle mutants first. Later passes relax subtlety, then the
    # quota (some types, e.g. wrong_boundary, have fewer test-failing mutants than the quota).
    counts, quota, chosen = Counter(), -(-n_buggy // len(BUG_TYPES)), {}
    for subtle_only, cap in ((True, quota), (False, quota), (True, n_buggy), (False, n_buggy)):
        for p in pool[n_correct:]:
            if len(chosen) == n_buggy or p["item_id"] in chosen:
                continue
            ms = [m for m in by_item[p["item_id"]] if m["subtle"] or not subtle_only]
            open_kinds = sorted({m["bug_type"] for m in ms if counts[m["bug_type"]] < cap})
            if open_kinds:
                kind = min(open_kinds, key=lambda t: (counts[t], rng.random()))
                counts[kind] += 1
                chosen[p["item_id"]] = (p, rng.choice([m for m in ms if m["bug_type"] == kind]))
    if len(chosen) < n_buggy:
        raise SystemExit(f"only {len(chosen)} buggy items found (bug types {dict(counts)})")
    for p, m in chosen.values():
        line, old, new = changed_line(p["code"], m["code"])
        items.append(dict(p, code=m["code"], reference_code=p["code"], is_buggy=True, bug_type=m["bug_type"],
                          mutation=m["op"], original_line_text=old, focus_line=line, focus_line_text=new,
                          n_tests_failed=p["n_tests"] - m["n_passed"],
                          test_error=m["test_error"], subtle=m["subtle"]))
    rng.shuffle(items)  # so any prefix of the file is a fair subset for pilots
    for it in items:
        it["test_code"] = it.pop("tests")
    write_jsonl(DATA / "items.jsonl", items)
    with open(DATA / "items.csv", "w", newline="") as f:  # what Angel's simulator reads
        w = csv.DictWriter(f, ["item_id", "source", "is_buggy", "bug_type"])
        w.writeheader()
        w.writerows({k: it[k] for k in w.fieldnames} for it in items)
    print(f"wrote {len(items)} items: bug types {dict(counts)}, "
          f"sources {dict(Counter((i['source'], i['is_buggy']) for i in items))}")
    return items


def self_check():
    """Re-run every item's tests: buggy items must fail >= 1 test, controls must pass all."""
    from .common import read_jsonl
    items = read_jsonl(DATA / "items.jsonl")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda it: run_tests(it["code"], it["test_code"]), items))
    bad = [it["item_id"] for it, r in zip(items, res) if passes(r) == it["is_buggy"]]
    assert not bad, f"items whose test outcome contradicts is_buggy: {bad}"
    print(f"self-check ok: {len(items)} items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-buggy", type=int, default=300)
    ap.add_argument("--n-correct", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only re-verify data/items.jsonl")
    a = ap.parse_args()
    if not a.check:
        build(a.n_buggy, a.n_correct, a.seed)
    self_check()
