"""Phase 8: malicious inputs against the API are blocked (revision Phase 8.3, AUDIT S1-S7).

Each attack writes a sentinel file or reads a secret if it succeeds; the tests check both the HTTP answer
and that the side effect did not happen.
"""
import io
import json
import time

import pytest

flask = pytest.importorskip("flask")

import apis  # noqa: E402
from service.config import Settings, settings_from_env  # noqa: E402
from service.expressions import ExpressionError, compile_expression, make_factory, parse_expression  # noqa: E402
from service.storage import LocalStore  # noqa: E402


def ui_code(name, params_lines, body, params="[]"):
    """What the Flutter UI posts (debias/lib/utilities/api_service.dart:102, schema_creator_view.dart:525-548):
    params_lines end with a newline; without parameters the UI sends '    \\n'."""
    params_lines = params_lines or "    \n"
    code = (f'@register_value_func("{name}")\ndef {name}_factory(params=None):\n    """\n    doc\n    """\n'
            f'    {params_lines}    return {body}\n\n\n{name}_factory._params = {params}')
    compile(code, "<ui>", "exec")             # parses: rejections below are by the checker, not a SyntaxError
    return code


@pytest.fixture
def make_client(tmp_path):
    def make(**kw):
        app = apis.create_app(Settings(**kw), LocalStore(tmp_path / "store"))
        return app.test_client()
    yield make
    apis.refresh_value_functions(LocalStore(tmp_path / "empty"), Settings())   # leave only built-ins


@pytest.fixture
def client(make_client):
    return make_client()


def attacks(sentinel):
    s = str(sentinel)
    return {
        "os.system": f"__import__('os').system('touch {s}')",
        "file read": "open('/etc/passwd').read()",
        "file write": f"open('{s}', 'w').write('x')",
        "network": "__import__('urllib.request').request.urlopen('http://example.com').read()",
        "socket": "__import__('socket').create_connection(('example.com', 80))",
        "attribute escape": "i.__class__.__mro__[1].__subclasses__()",
        "numpy escape": "np.os.system('id')",
        "getattr": "getattr(np, 'os')",
        "lambda": "(lambda: 1)()",
        "comprehension": "[x for x in range(10**9)]",
        "infinite loop": "[i for i in iter(int, 1)]",
        "memory: size argument": "np.random.normal(0, 1, 10**9)",
        "memory: size keyword": "np.random.normal(0, 1, size=10**9)",
        "string bomb": "'a' * 10**9",
        "walrus": "(x := 1)",
        "f-string": "f'{i}'",
    }


@pytest.mark.parametrize("attack", list(attacks("x")))
def test_expression_attacks_rejected(attack, tmp_path):
    sentinel = tmp_path / "pwned"
    with pytest.raises(ExpressionError):
        parse_expression(attacks(sentinel)[attack], {"i"})
    assert not sentinel.exists()


@pytest.mark.parametrize("attack", list(attacks("x")))
def test_api_rejects_attacks_in_ui_code_and_in_specs(client, attack, tmp_path):
    sentinel = tmp_path / "pwned"
    expr = attacks(sentinel)[attack]
    r = client.post("/submit-value-function-api", json={"code": ui_code("evil", "", f"lambda i: {expr}")})
    assert r.status_code == 400 and r.json["success"] is False
    r = client.post("/submit-value-function-api",
                    json={"name": "evil", "doc": "d", "params": {}, "expression": expr})
    assert r.status_code == 400 and r.json["success"] is False
    assert not sentinel.exists()
    assert "evil" not in [f["name"] for f in client.get("/get-value-functions-api").json]


def test_original_audit_exploit_does_not_run(client, tmp_path):
    """AUDIT S1: code outside the template ran before any check. Now it is never executed."""
    sentinel = tmp_path / "pwned"
    code = (f"import os\nos.system('touch {sentinel}')\n" +
            ui_code("evil", "", "lambda i: i"))
    r = client.post("/submit-value-function-api", json={"code": code})
    assert r.status_code == 400
    hidden = ui_code("evil2", f"x = __import__('os').system('touch {sentinel}')\n", "lambda i: i")
    assert client.post("/submit-value-function-api", json={"code": hidden}).status_code == 400
    loop = ui_code("evil3", "while True:\n        pass\n", "lambda i: i")
    assert client.post("/submit-value-function-api", json={"code": loop}).status_code == 400
    assert not sentinel.exists()


def test_cpu_and_overflow_bounded():
    t = time.perf_counter()
    assert compile_expression("9**9**9**9", {})(1) == float("inf")
    assert compile_expression("1/0", {})(1) == float("inf")
    for bad in ["-" * 1999 + "1", "(" * 300 + "1" + ")" * 300, "1" + "0" * 400, "i+" * 1000 + "i"]:
        with pytest.raises(ExpressionError):
            parse_expression(bad, {"i"})
    assert time.perf_counter() - t < 2


def test_ui_template_accepted_and_evaluated_without_exec(client):
    code = ui_code("trend2", 'slope = params.get("slope", 2.0)\n    b = params.get("b", -1)\n',
                   "lambda i: b + slope * i + 0 * np.random.normal(0, 1)", '["slope", "b"]')
    r = client.post("/submit-value-function-api", json={"code": code})
    assert r.status_code == 200 and r.json["test_output"] == 9.0
    info = next(f for f in client.get("/get-value-functions-api").json if f["name"] == "trend2")
    assert info["params"] == ["slope", "b"] and info["defaults"] == {"slope": 2.0, "b": -1}
    assert client.post("/submit-value-function-api", json={"code": code}).status_code == 400   # duplicate
    builtin = ui_code("trend_noise", "", "lambda i: i")
    assert client.post("/submit-value-function-api", json={"code": builtin}).status_code == 400  # built-in name


def test_factory_rejects_bad_params():
    f = make_factory({"name": "f", "doc": "d", "params": {"a": 1}, "expression": "a * i"})
    assert f({"a": 3})(2) == 6.0
    with pytest.raises(ValueError):
        f({"zzz": 1})
    with pytest.raises(ValueError):
        f({"a": "__import__('os')"})


def test_legacy_exec_disabled_by_default_and_opt_in(make_client, tmp_path):
    sentinel = tmp_path / "ran"
    code = ui_code("legacy_fn", f"open({str(sentinel)!r}, 'w').write('x')\n", "lambda i: i", "[]")
    off = make_client()
    assert off.post("/submit-value-function-api", json={"code": code}).status_code == 400
    assert not sentinel.exists()
    on = make_client(legacy_code_exec=True)            # documented: restores arbitrary code execution
    r = on.post("/submit-value-function-api", json={"code": code})
    assert r.status_code == 200 and sentinel.exists()
    sentinel.unlink()
    make_client()                                      # restart without the flag: stored code is not loaded
    assert not sentinel.exists()


def test_detect_outliers_not_evaluated(client, tmp_path):
    """AUDIT S3: the form field went through eval()."""
    sentinel = tmp_path / "pwned"
    form = {"detect_outliers": f"__import__('os').system('touch {sentinel}')", "z_thresh": "3",
            "target_variables": "y", "target_variables_descriptions": "y", "features": "a",
            "features_descriptions": "a", "file": (io.BytesIO(b"a,y\n1,0\n"), "d.csv")}
    r = client.post("/data-bias-api", data=form, content_type="multipart/form-data")
    assert r.status_code == 400 and not sentinel.exists()


@pytest.mark.parametrize("path", ["/etc/passwd", "../apis.py", "generated/../../apis.py", "generated/..",
                                  "apis.py", "schemas/x.json", "generated\\..\\apis.py", "", "/generated/x.json",
                                  "generated/%2e%2e/apis.py"])
def test_no_arbitrary_file_read(client, path):
    """AUDIT S4: ?path= was any file on the server."""
    for route in ("/get-data-api", "/get-plot-api"):
        r = client.get(route, query_string={"path": path})
        assert r.status_code in (400, 404) and b"root:" not in r.data


@pytest.mark.parametrize("name", ["../x", "a/b", "..", "x" * 65, "", "a b", "../../etc/cron.d/x"])
def test_schema_name_traversal_rejected(client, name):
    """AUDIT S5: the schema name went into file paths."""
    r = client.post("/submit-data-schema-api", json={"name": name, "schema": {"a": {"type": "int"}}})
    assert r.status_code == 400


@pytest.mark.parametrize("schema,n_rows", [
    ({"a": {"type": "int", "min": 0, "max": 5}}, 10**12),
    ({"a": {"type": "int", "min": 0, "max": 5}}, -1),
    ({"a": {"type": "string", "length": 10**9}}, 1),
    ({"a": {"type": "pattern", "pattern": "a{999999999}"}}, 1),
    ({"a": {"type": "pattern", "pattern": "(a{1000}){1000}"}}, 1),
    ({"a": {"type": "pattern", "pattern": "((a*)*)*"}}, 1),
    ({"u": {"type": "id"}, "t": {"type": "timeseries", "start": "2024-01-01", "periods": 10**9, "freq": "s",
                                 "value_func": {"name": "oscillate", "params": {}}, "entity_col": "u"}}, 1),
])
def test_memory_bombs_rejected(client, schema, n_rows):
    r = client.post("/submit-data-schema-api", json={"name": "bomb", "schema": schema})
    if n_rows == 1:
        assert r.status_code == 400
        return
    assert r.status_code == 200                       # the schema itself is fine; the request size is not
    r = client.post("/generate-synthetic-data-api", json={"schema_name": "bomb", "n_rows": n_rows})
    assert r.status_code == 400


def test_upload_size_limit(make_client):
    c = make_client(max_upload_mb=1)
    big = io.BytesIO(b"a,b\n" + b"1,2\n" * 400_000)
    r = c.post("/data-bias-api", data={"file": (big, "d.csv")}, content_type="multipart/form-data")
    assert r.status_code == 413


def test_api_key_and_cors(make_client):
    c = make_client(api_key="s3cret")
    assert c.get("/get-value-functions-api").status_code == 401
    assert c.get("/get-value-functions-api", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/get-value-functions-api", headers={"X-API-Key": "s3cret"}).status_code == 200
    c = make_client()
    evil = c.get("/get-value-functions-api", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in evil.headers
    ok = c.get("/get-value-functions-api", headers={"Origin": "http://localhost:8080"})
    assert ok.headers.get("Access-Control-Allow-Origin") == "http://localhost:8080"


def test_safe_defaults():
    s = settings_from_env({})
    assert s.host == "127.0.0.1" and s.debug is False and s.legacy_code_exec is False
    assert settings_from_env({"DEBIAS_ENABLE_LEGACY_CODE_EXEC": "1"}).legacy_code_exec is True


def test_generated_plot_is_sandboxed(client):
    schema = {"a": {"type": "int", "min": 0, "max": 3}, "<script>alert(1)</script>": {"type": "bool"}}
    assert client.post("/submit-data-schema-api", json={"name": "s1", "schema": schema}).status_code == 200
    r = client.post("/generate-synthetic-data-api", json={"schema_name": "s1", "n_rows": 20})
    assert r.status_code == 200
    plot = client.get("/get-plot-api", query_string={"path": r.json["html_file_path"]})
    assert plot.status_code == 200 and plot.headers["Content-Security-Policy"] == "sandbox allow-scripts"
    data = client.get("/get-data-api", query_string={"path": r.json["generated_data_path"]})
    assert data.status_code == 200 and len(data.json["data"]) == 20


def test_every_builtin_value_function_is_expressible():
    """The shipped value functions all fit the declarative language and give the same numbers."""
    import ast
    import inspect

    import numpy as np

    import bias_mitigation.synthetic_data.value_functions as vf
    from service.expressions import spec_from_ui_template

    src = inspect.getsource(vf)
    tree = ast.parse(src)
    for name, factory in apis.BUILTIN_VALUE_FUNCS.items():
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == factory.__name__)
        lines = src.splitlines()[fn.decorator_list[0].lineno - 1: fn.end_lineno]   # decorator + function
        spec = spec_from_ui_template("\n".join(lines))
        mine = make_factory(spec)
        for i in (0, 7, 1234):
            np.random.seed(i)
            a = factory({})(i)
            np.random.seed(i)
            b = mine({})(i)
            assert np.isclose(a, b, rtol=1e-12, atol=1e-9), (name, i, a, b)
