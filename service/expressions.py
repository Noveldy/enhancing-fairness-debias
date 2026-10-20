"""Declarative value functions: a small, allow-listed arithmetic language (Phase 8, D4 option a).

A user-defined value function is data, not code:

    {"name": "trend_noise2", "doc": "Linear trend with noise.",
     "params": {"slope": 1.0, "intercept": 0.0},
     "expression": "intercept + slope * i + np.random.normal(0, 1)"}

The expression is parsed with `ast.parse(mode="eval")` and checked node by node against an allow-list:
numbers, the index `i`, the declared parameter names, arithmetic, comparisons, `a if cond else b`, and calls
to the functions in FUNCTIONS. Attribute access is accepted only as a complete dotted name from that list
(`np.sin`, `np.random.normal`, `math.pi`); the dotted name is looked up in our own table, never with
getattr on a real object. Evaluation walks the checked tree with our own interpreter. Nothing is passed to
eval, exec or compile, so no Python builtins, imports, attributes or files are reachable from an expression.

Resource limits: expression length, node count, nesting depth, call arity (no `size=` arguments, so a call
returns one number); all arithmetic is numpy float64 with floating-point errors silenced, so `9**9**9**9`
gives inf instead of building a huge integer, and division by zero gives inf/nan instead of raising.
"""
from __future__ import annotations

import ast
import inspect
import math
import random
import re

import numpy as np

MAX_CHARS = 2000
MAX_NODES = 300
MAX_DEPTH = 40
MAX_PARAMS = 32
MAX_DOC_CHARS = 2000
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ExpressionError(ValueError):
    """The expression or value-function spec is not allowed."""


def _f(x):
    return np.float64(x)


# -- allow-listed functions: fixed signatures, scalar in, scalar out -------------------------------------
def _normal(loc=0.0, scale=1.0):
    return _f(np.random.normal(loc, scale))


def _uniform(low=0.0, high=1.0):
    return _f(np.random.uniform(low, high))


def _np_random():
    return _f(np.random.random())


def _poisson(lam=1.0):
    return _f(np.random.poisson(lam))


def _exponential(scale=1.0):
    return _f(np.random.exponential(scale))


def _randint(low, high):            # numpy convention: high exclusive
    return _f(np.random.randint(int(low), int(high)))


def _py_uniform(a, b):
    return _f(random.uniform(a, b))


def _py_gauss(mu=0.0, sigma=1.0):
    return _f(random.gauss(mu, sigma))


def _py_random():
    return _f(random.random())


def _py_randint(a, b):              # Python convention: b inclusive
    return _f(random.randint(int(a), int(b)))


def _unary(fn):
    def call(x):
        return _f(fn(x))
    return call


def _min(a, b):
    return _f(min(a, b))


def _max(a, b):
    return _f(max(a, b))


def _clip(x, lo, hi):
    return _f(np.clip(x, lo, hi))


_MATH = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos,
         "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp,
         "log": np.log, "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt, "abs": np.abs,
         "floor": np.floor, "ceil": np.ceil, "round": np.round}

FUNCTIONS: dict[str, callable] = {}
for _name, _fn in _MATH.items():
    for _prefix in ("", "np.", "numpy.", "math."):
        if _prefix == "math." and _name in ("arcsin", "arccos", "arctan", "abs", "round"):
            continue
        FUNCTIONS[_prefix + _name] = _unary(_fn)
FUNCTIONS.update({"math.asin": _unary(np.arcsin), "math.acos": _unary(np.arccos),
                  "math.atan": _unary(np.arctan), "math.fabs": _unary(np.abs)})
FUNCTIONS.update({"min": _min, "max": _max, "np.minimum": _min, "np.maximum": _max, "np.clip": _clip,
                  "clip": _clip,
                  "np.random.normal": _normal, "numpy.random.normal": _normal,
                  "np.random.uniform": _uniform, "numpy.random.uniform": _uniform,
                  "np.random.random": _np_random, "np.random.rand": _np_random,
                  "np.random.poisson": _poisson, "np.random.exponential": _exponential,
                  "np.random.randint": _randint,
                  "random.uniform": _py_uniform, "random.gauss": _py_gauss, "random.normalvariate": _py_gauss,
                  "random.random": _py_random, "random.randint": _py_randint})

CONSTANTS = {"pi": math.pi, "e": math.e, "np.pi": math.pi, "np.e": math.e, "numpy.pi": math.pi,
             "math.pi": math.pi, "math.e": math.e, "math.tau": math.tau, "np.nan": math.nan, "nan": math.nan,
             "np.inf": math.inf, "inf": math.inf}

RESERVED = {name for name in list(FUNCTIONS) + list(CONSTANTS) if "." not in name} | {
    "i", "np", "numpy", "math", "random", "params", "True", "False", "None"}

_SIGS = {name: inspect.signature(fn) for name, fn in FUNCTIONS.items()}

_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.true_divide,
           ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
           ast.Eq: np.equal, ast.NotEq: np.not_equal}


def _dotted(node):
    """'np.random.normal' for an Attribute/Name chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _depth(node, d=0):
    kids = list(ast.iter_child_nodes(node))
    if d > MAX_DEPTH:
        return d
    return max([d] + [_depth(k, d + 1) for k in kids])


def parse_expression(text: str, variables) -> ast.Expression:
    """Parse and check an expression; return the checked tree or raise ExpressionError."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression must be a non-empty string")
    if len(text) > MAX_CHARS:
        raise ExpressionError(f"expression longer than {MAX_CHARS} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise ExpressionError(f"cannot parse expression: {e.__class__.__name__}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ExpressionError(f"expression has more than {MAX_NODES} nodes")
    if _depth(tree) > MAX_DEPTH:
        raise ExpressionError(f"expression nested deeper than {MAX_DEPTH}")
    variables = set(variables)
    _check(tree.body, variables)
    return tree


def _check(node, variables):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"constant {node.value!r} is not a number")
        try:
            float(node.value)
        except OverflowError:
            raise ExpressionError("numeric constant too large") from None
        return
    if isinstance(node, ast.Name):
        if node.id in variables or node.id in CONSTANTS:
            return
        raise ExpressionError(f"unknown name '{node.id}' (allowed: i, the declared parameters, constants)")
    if isinstance(node, ast.Attribute):
        name = _dotted(node)
        if name in CONSTANTS:
            return
        raise ExpressionError(f"attribute access '{name or ast.unparse(node)}' is not allowed")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        _check(node.left, variables)
        _check(node.right, variables)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise ExpressionError(f"operator {type(node.op).__name__} is not allowed")
        _check(node.operand, variables)
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v, variables)
        return
    if isinstance(node, ast.Compare):
        if not all(type(op) in _CMPOPS for op in node.ops):
            raise ExpressionError("only <, <=, >, >=, ==, != comparisons are allowed")
        for v in [node.left, *node.comparators]:
            _check(v, variables)
        return
    if isinstance(node, ast.IfExp):
        for v in (node.test, node.body, node.orelse):
            _check(v, variables)
        return
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name not in FUNCTIONS:
            raise ExpressionError(f"function '{name or ast.unparse(node.func)}' is not allowed")
        if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
            raise ExpressionError("*args / **kwargs are not allowed")
        try:
            _SIGS[name].bind(*node.args, **{k.arg: k.value for k in node.keywords})
        except TypeError as e:
            raise ExpressionError(f"bad arguments for {name}: {e}") from None
        for a in [*node.args, *(k.value for k in node.keywords)]:
            _check(a, variables)
        return
    raise ExpressionError(f"{type(node).__name__} is not allowed in an expression")


def _eval(node, env):
    if isinstance(node, ast.Constant):
        return _f(node.value)
    if isinstance(node, ast.Name):
        return env[node.id] if node.id in env else _f(CONSTANTS[node.id])
    if isinstance(node, ast.Attribute):
        return _f(CONSTANTS[_dotted(node)])
    if isinstance(node, ast.BinOp):
        return _f(_BINOPS[type(node.op)](_eval(node.left, env), _eval(node.right, env)))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Not):
            return _f(not v)
        return v
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, env) for v in node.values)
        return _f(all(vals) if isinstance(node.op, ast.And) else any(vals))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if not _CMPOPS[type(op)](left, right):
                return _f(0.0)
            left = right
        return _f(1.0)
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    if isinstance(node, ast.Call):
        fn = FUNCTIONS[_dotted(node.func)]
        args = [_eval(a, env) for a in node.args]
        kwargs = {k.arg: _eval(k.value, env) for k in node.keywords}
        try:
            return _f(fn(*args, **kwargs))
        except (ValueError, OverflowError, ZeroDivisionError):   # e.g. normal(0, -1), randint(5, 5)
            return _f(math.nan)
    raise ExpressionError(f"{type(node).__name__} is not allowed")   # unreachable after _check


def compile_expression(text: str, params) -> callable:
    """Return f(i, **params) -> float for a checked expression."""
    tree = parse_expression(text, set(params) | {"i"})

    def evaluate(i, **values):
        env = {k: _f(v) for k, v in values.items()}
        env["i"] = _f(i)
        with np.errstate(all="ignore"):
            return float(_eval(tree.body, env))
    return evaluate


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_spec(spec) -> dict:
    """Check a value-function spec; return a normalised copy or raise ExpressionError."""
    if not isinstance(spec, dict):
        raise ExpressionError("value function must be a JSON object")
    extra = set(spec) - {"name", "doc", "params", "expression"}
    if extra:
        raise ExpressionError(f"unknown keys {sorted(extra)}")
    name, doc = spec.get("name"), spec.get("doc", "")
    params, expression = spec.get("params", {}) or {}, spec.get("expression")
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ExpressionError("name must match [A-Za-z][A-Za-z0-9_]{0,63}")
    if not isinstance(doc, str) or not doc.strip() or len(doc) > MAX_DOC_CHARS:
        raise ExpressionError(f"doc must be a non-empty string of at most {MAX_DOC_CHARS} characters")
    if not isinstance(params, dict) or len(params) > MAX_PARAMS:
        raise ExpressionError(f"params must be an object with at most {MAX_PARAMS} entries")
    for k, v in params.items():
        if not isinstance(k, str) or not NAME_RE.match(k) or k in RESERVED:
            raise ExpressionError(f"parameter name '{k}' is not allowed")
        if not _number(v):
            raise ExpressionError(f"default of parameter '{k}' must be a finite number")
    parse_expression(expression, set(params) | {"i"})
    return {"name": name, "doc": doc.strip(), "params": dict(params), "expression": expression.strip()}


def make_factory(spec: dict):
    """A factory with the same interface as the built-in ones: factory(params) -> f(i)."""
    spec = validate_spec(spec)
    evaluate = compile_expression(spec["expression"], spec["params"])
    defaults = spec["params"]

    def factory(params=None):
        params = dict(params or {})
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)} for value function '{spec['name']}'")
        bad = [k for k, v in params.items() if not _number(v)]
        if bad:
            raise ValueError(f"parameters {bad} must be finite numbers")
        values = {**defaults, **params}
        return lambda i: evaluate(i, **values)

    factory.__doc__ = spec["doc"]
    factory.__name__ = f"{spec['name']}_factory"
    factory._params = list(defaults)
    factory._spec = spec
    return factory


# -- the code template the Flutter UI posts (debias/lib/utilities/api_service.dart:102) ------------------
def spec_from_ui_template(code: str) -> dict:
    """Read the UI's generated Python *without running it* and turn it into a spec.

    Accepted shape (anything else is rejected):

        @register_value_func("NAME")
        def NAME_factory(params=None):
            \"\"\"doc\"\"\"
            p = params.get("p", <number>)        # zero or more
            return lambda i: <expression>
        NAME_factory._params = ["p", ...]        # optional
    """
    if not isinstance(code, str) or len(code) > 4 * MAX_CHARS:
        raise ExpressionError("code missing or too long")
    try:
        mod = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        raise ExpressionError(f"cannot parse code: {e.__class__.__name__}") from None
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    rest = [n for n in mod.body if not isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ExpressionError("expected exactly one factory function")
    fn = funcs[0]
    if not fn.name.endswith("_factory"):
        raise ExpressionError("the function name must end with '_factory'")
    if len(fn.decorator_list) != 1:
        raise ExpressionError("expected the @register_value_func decorator")
    dec = fn.decorator_list[0]
    if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "register_value_func"
            and not dec.keywords and len(dec.args) <= 1
            and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in dec.args)):
        raise ExpressionError("expected @register_value_func(\"name\")")
    name = dec.args[0].value if dec.args else fn.name[: -len("_factory")]
    a = fn.args
    if (a.posonlyargs or a.vararg or a.kwarg or a.kwonlyargs or [x.arg for x in a.args] != ["params"]
            or len(a.defaults) > 1 or (a.defaults and not (isinstance(a.defaults[0], ast.Constant)
                                                          and a.defaults[0].value is None))):
        raise ExpressionError("the factory must be declared as NAME_factory(params=None)")
    body = list(fn.body)
    doc = ""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc = body.pop(0).value.value
    params = {}
    while body and isinstance(body[0], ast.Assign):
        st = body.pop(0)
        if not (len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
            raise ExpressionError("parameter lines must look like  p = params.get(\"p\", <number>)")
        target, call = st.targets[0].id, st.value
        if not (isinstance(call, ast.Call) and _dotted(call.func) == "params.get" and not call.keywords
                and len(call.args) == 2 and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == target):
            raise ExpressionError("parameter lines must look like  p = params.get(\"p\", <number>)")
        try:
            default = ast.literal_eval(call.args[1])
        except ValueError:
            raise ExpressionError(f"default of '{target}' must be a number") from None
        params[target] = default
    if len(body) != 1 or not isinstance(body[0], ast.Return) or not isinstance(body[0].value, ast.Lambda):
        raise ExpressionError("the factory must end with  return lambda i: <expression>")
    lam = body[0].value
    la = lam.args
    if la.posonlyargs or la.vararg or la.kwarg or la.kwonlyargs or la.defaults or [x.arg for x in la.args] != ["i"]:
        raise ExpressionError("the returned function must be  lambda i: ...")
    for st in rest:                         # only  NAME_factory._params = ["a", "b"]
        ok = (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute)
              and st.targets[0].attr == "_params" and isinstance(st.targets[0].value, ast.Name)
              and st.targets[0].value.id == fn.name and isinstance(st.value, ast.List)
              and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in st.value.elts))
        if not ok:
            raise ExpressionError("only  NAME_factory._params = [\"p\", ...]  may follow the function")
    spec = {"name": name, "doc": doc or f"User-defined value function {name}.", "params": params,
            "expression": ast.unparse(lam.body)}
    return validate_spec(spec)
