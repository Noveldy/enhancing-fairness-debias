"""Size limits for synthetic-data requests (Phase 8: resource exhaustion through schemas).

Without these, one request could ask for billions of rows (`n_rows`), a timeseries with billions of
`periods`, a 10^9-character `string` column, or a regex such as `a{999999999}` for a `pattern` column
(rstr.xeger builds the whole string). The checks run before anything is generated.
"""
from __future__ import annotations

import re

try:                                           # Python >= 3.11
    import re._constants as _C
    import re._parser as _P
except ImportError:                            # pragma: no cover
    import sre_constants as _C
    import sre_parse as _P

MAX_COLUMNS = 500
MAX_STRING_LENGTH = 10_000
MAX_PATTERN_CHARS = 500
MAX_PATTERN_WIDTH = 10_000
MAX_CATEGORIES = 10_000
RSTR_STAR_LIMIT = 100                          # rstr's repeat count for * and +
SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class LimitError(ValueError):
    pass


def _width(parsed) -> int:
    total = 0
    for op, av in parsed:
        if op in (_C.LITERAL, _C.NOT_LITERAL, _C.ANY, _C.IN, _C.CATEGORY):
            total += 1
        elif op == _C.SUBPATTERN:
            total += _width(av[-1])
        elif op == _C.BRANCH:
            total += max((_width(p) for p in av[1]), default=0)
        elif op in (_C.MAX_REPEAT, _C.MIN_REPEAT) or op == getattr(_C, "POSSESSIVE_REPEAT", None):
            lo, hi, sub = av
            hi = RSTR_STAR_LIMIT if hi == _C.MAXREPEAT else hi
            total += max(hi, lo) * _width(sub)
        elif op == _C.GROUPREF:
            total += MAX_PATTERN_WIDTH            # a back-reference repeats a group: count it as the maximum
        elif op in (_C.ASSERT, _C.ASSERT_NOT):
            total += _width(av[1])
        if total > MAX_PATTERN_WIDTH:
            return total
    return total


def pattern_width(pattern: str) -> int:
    if not isinstance(pattern, str) or len(pattern) > MAX_PATTERN_CHARS:
        raise LimitError(f"pattern must be a string of at most {MAX_PATTERN_CHARS} characters")
    try:
        return _width(_P.parse(pattern))
    except (re.error, RecursionError, OverflowError) as e:
        raise LimitError(f"invalid pattern: {e}") from None


def _int(v, what, lo=0, hi=None):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise LimitError(f"{what} must be an integer in [{lo}, {hi}]")
    return v


def check_schema_name(name):
    if not isinstance(name, str) or not SCHEMA_NAME_RE.match(name):
        raise LimitError("schema name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
    return name


def check_request(schema: dict, n_rows, max_rows: int) -> int:
    """Raise LimitError if generating `n_rows` from `schema` could exceed the limits; return rows produced."""
    if not isinstance(schema, dict) or not schema:
        raise LimitError("schema must be a non-empty object")
    if len(schema) > MAX_COLUMNS:
        raise LimitError(f"schema has more than {MAX_COLUMNS} columns")
    n_rows = _int(n_rows, "n_rows", 1, max_rows)
    periods_total = 0
    for col, d in schema.items():
        if not isinstance(d, dict):
            raise LimitError(f"definition of '{col}' must be an object")
        t = d.get("type")
        if t == "string":
            _int(d.get("length", 10), f"'{col}'.length", 0, MAX_STRING_LENGTH)
        elif t == "pattern":
            if pattern_width(d.get("pattern")) > MAX_PATTERN_WIDTH:
                raise LimitError(f"pattern of '{col}' can produce strings longer than {MAX_PATTERN_WIDTH}")
        elif t == "category":
            cats = d.get("categories")
            if not isinstance(cats, list) or len(cats) > MAX_CATEGORIES:
                raise LimitError(f"'{col}'.categories must be a list of at most {MAX_CATEGORIES}")
        elif t == "timeseries":
            periods_total += _int(d.get("periods"), f"'{col}'.periods", 1, max_rows)
    rows = n_rows * max(1, periods_total)
    if rows > max_rows:
        raise LimitError(f"request would produce {rows:,} rows; the limit is {max_rows:,} (DEBIAS_MAX_ROWS)")
    return rows
