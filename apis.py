"""DEBIAS REST API (revised in Phase 8; the original is in legacy/original/apis.py).

Security changes (README.md "API security", docs/AUDIT.md S1-S9):
- Value functions are declarative expressions (service/expressions.py), stored as data with integrity
  checks (service/value_store.py). The code-accepting endpoint reads the UI's code template without running
  it; arbitrary code runs only if DEBIAS_ENABLE_LEGACY_CODE_EXEC=1 (off by default).
- No eval() of form fields; file access only through validated storage keys (no client-supplied paths);
  schema names validated; size limits on uploads and generated data.
- Local storage by default, MinIO when DEBIAS_MINIO_ENDPOINT is set (service/storage.py).
- Binds to 127.0.0.1 with the debugger off; optional API key (DEBIAS_API_KEY); CORS limited to localhost
  origins unless DEBIAS_CORS_ORIGINS says otherwise.
"""
import hmac
import io
import json
import logging
import random
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

import bias_mitigation.synthetic_data.value_functions as vf
from bias_mitigation.synthetic_data.synthetic_data_generator import (AdvancedSyntheticDataGenerator, SyntheticIO,
                                                                     TimeSeriesSyntheticGenerator)
from service import value_store
from service.config import Settings, settings_from_env
from service.expressions import NAME_RE, ExpressionError, make_factory, spec_from_ui_template, validate_spec
from service.limits import LimitError, check_request, check_schema_name
from service.storage import NotFound, StorageError, check_key, store_from_env

log = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {'csv', 'json'}
BUILTIN_VALUE_FUNCS = dict(vf.VALUE_FUNC_FACTORY_REGISTRY)     # shipped with the code, trusted
GENERATED_PREFIX = "generated/"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_bool(value):
    v = str(value).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError("must be true or false")


def refresh_value_functions(store, settings):
    """Built-in functions plus the stored user functions that pass the integrity checks."""
    user, rejected = value_store.load_specs(store, settings.hmac_key)
    if settings.legacy_code_exec:
        legacy, bad = value_store.load_legacy_code(store, _legacy_namespace, settings.hmac_key)
        user.update(legacy)
        rejected += bad
    registry = vf.VALUE_FUNC_FACTORY_REGISTRY          # the generator holds a reference to this dict
    registry.clear()
    registry.update(BUILTIN_VALUE_FUNCS)
    registry.update({k: v for k, v in user.items() if k not in BUILTIN_VALUE_FUNCS})
    return rejected


def _legacy_namespace():
    registered = {}

    def register_value_func(name=None):
        def deco(func):
            registered[name or func.__name__] = func
            return func
        return deco
    return {"np": np, "random": random, "register_value_func": register_value_func, "__registered__": registered}


def create_app(settings: Settings = None, store=None):
    settings = settings or settings_from_env()
    store = store or store_from_env()
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = settings.max_upload_mb * 1024 * 1024
    app.config['DEBIAS_SETTINGS'] = settings
    app.config['DEBIAS_STORE'] = store
    CORS(app, origins=list(settings.cors_origins))
    refresh_value_functions(store, settings)

    def fail(msg, status=400):
        return jsonify({"success": False, "error": msg}), status

    @app.before_request
    def require_api_key():
        if settings.api_key and request.method != "OPTIONS":
            given = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(given.encode(), settings.api_key.encode()):
                return fail("missing or wrong X-API-Key", 401)

    # this is the data bias API
    @app.route('/data-bias-api', methods=['POST'])
    def data_bias_api():
        if 'file' not in request.files:
            return Response('No file part', status=400, mimetype="text/plain")
        file = request.files['file']
        if file.filename == '':
            return Response('No file was selected', status=400, mimetype="text/plain")
        if not (file and allowed_file(file.filename)):
            return Response('Only .csv and .json files are accepted', status=400, mimetype="text/plain")
        filename = secure_filename(file.filename)
        try:
            detect_outliers = parse_bool(request.form['detect_outliers'])   # was eval(): AUDIT S3
            z_thresh = int(request.form['z_thresh'])
            target_variables = request.form['target_variables'].split(',')
            target_variables_descriptions = request.form['target_variables_descriptions'].split(',')
            features = request.form['features'].split(',')
            features_descriptions = request.form['features_descriptions'].split(',')
        except (KeyError, ValueError) as e:
            return Response(f'Invalid form field: {e}', status=400, mimetype="text/plain")
        raw = file.read()
        # keep the upload under its own key, so uploads of different users do not overwrite each other
        store.put(f"uploads/{uuid.uuid4().hex}_{filename}", raw, file.mimetype or "application/octet-stream")
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(io.BytesIO(raw))
            else:
                data = json.loads(raw)
                if 'data' in data:
                    data = data['data']
                data = pd.DataFrame(json.loads(data))
        except Exception as e:
            return Response(f'Cannot read the file: {e}', status=400, mimetype="text/plain")
        from main import check_dataset_bias        # heavy (transformers, darts): load only when used
        Path('results').mkdir(exist_ok=True)
        with open(Path('results/logs.txt'), 'w') as log_file:
            log_file.write('')
        try:
            filename = filename.split('.')[0]
            mechanism_response = check_dataset_bias(data, filename, detect_outliers, z_thresh, target_variables,
                                                    target_variables_descriptions, features, features_descriptions)
            debias_ops_log = mechanism_response.pop('debias_ops_log')
            debias_ops_log = '\n'.join(dict.fromkeys(debias_ops_log.split('\n')))   # drop duplicate lines
            mechanism_response['debias_ops_log'] = debias_ops_log
            return Response(json.dumps(mechanism_response), status=200, mimetype="application/json")
        except Exception as e:
            return Response(str(e), status=500, mimetype="text/plain")

    # this is the API for retrieving all the value functions
    @app.route('/get-value-functions-api', methods=['GET'])
    def list_value_functions():
        refresh_value_functions(store, settings)
        functions_info = []
        for name, func in vf.VALUE_FUNC_FACTORY_REGISTRY.items():
            info = {"name": name,
                    "doc": func.__doc__.strip() if func.__doc__ else "No description available.",
                    "params": list(getattr(func, "_params", []))}
            if hasattr(func, "_spec"):
                info["expression"] = func._spec["expression"]
                info["defaults"] = func._spec["params"]
            functions_info.append(info)
        return jsonify(functions_info)

    # this is the API for submitting a new value function
    @app.route('/submit-value-function-api', methods=['POST'])
    def submit_value_function():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return fail("Request body must be a JSON object")
        try:
            if "code" in body:                       # the UI's code template, read without running it
                if not body["code"]:
                    return fail("No code provided")
                try:
                    spec = spec_from_ui_template(body["code"])
                except ExpressionError as e:
                    if settings.legacy_code_exec:
                        return _legacy_submit(body["code"])
                    return fail(f"Not accepted: {e}. Value functions are expressions over i and the declared "
                                "parameters, using arithmetic and the allowed math/random functions.")
            else:
                spec = validate_spec(body.get("spec", body))
            spec = value_store.save_spec(store, spec, settings.hmac_key, reserved=BUILTIN_VALUE_FUNCS)
        except ExpressionError as e:
            return fail(str(e))
        refresh_value_functions(store, settings)
        test_output = make_factory(spec)({})(5)
        return jsonify({"success": True,
                        "message": f"Function '{spec['name']}' successfully validated and saved.",
                        "test_output": None if np.isnan(test_output) else test_output})

    def _legacy_submit(code):
        """Original behaviour (exec of user code), only with DEBIAS_ENABLE_LEGACY_CODE_EXEC=1."""
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            return fail(f"Syntax error: {e}")
        if "@register_value_func" not in code:
            return fail("Missing @register_value_func decorator.")
        ns = _legacy_namespace()
        try:
            exec(code, ns)                                        # noqa: S102  ARBITRARY CODE EXECUTION
            items = [(k, v) for k, v in ns.items() if callable(v) and k.endswith("_factory")]
            if not items:
                return fail("No valid factory function found (must end with '_factory').")
            func_name, factory = items[0]
            if not factory.__doc__ or not factory.__doc__.strip():
                return fail(f"Factory '{func_name}' is missing a docstring.")
            if not isinstance(getattr(factory, "_params", None), list):
                return fail(f"Factory '{func_name}' must define a `_params` list.")
            test_output = factory({})(5)
            name = next(iter(ns["__registered__"]), func_name)
            if not NAME_RE.match(name):
                return fail("Function name must match [A-Za-z][A-Za-z0-9_]{0,63}.")
            if name in vf.VALUE_FUNC_FACTORY_REGISTRY:
                return fail(f"Function '{name}' already exists.")
            value_store.save_legacy_code(store, name, code, settings.hmac_key)
        except (ExpressionError, StorageError) as e:
            return fail(str(e))
        except Exception as e:
            return fail(f"Function test failed: {e}")
        refresh_value_functions(store, settings)
        return jsonify({"success": True, "message": f"Function '{name}' saved (legacy code path).",
                        "test_output": test_output})

    def _load_schema(name):
        return json.loads(store.get(f"schemas/{check_schema_name(name)}.json"))

    # this is the API for submitting a data schema
    @app.route('/submit-data-schema-api', methods=['POST'])
    def submit_data_schema():
        data = request.get_json(silent=True)
        if not data or "name" not in data or "schema" not in data:
            return fail("Request must contain 'name' and 'schema' fields.")
        name, schema = data["name"], data["schema"]
        if not isinstance(name, str) or not isinstance(schema, dict):
            return fail("'name' must be a string and 'schema' must be a dictionary.")
        try:
            check_schema_name(name)                                   # was used in file paths: AUDIT S5
            check_request(schema, 1, settings.max_rows)
        except LimitError as e:
            return fail(str(e))
        for field, config in schema.items():
            if "type" not in config:
                return fail(f"Missing 'type' for field '{field}'.")
            if config["type"] == "timeseries":
                missing = {"start", "periods", "freq", "value_func", "entity_col"} - config.keys()
                if missing:
                    return fail(f"Missing keys {missing} in timeseries field '{field}'.")
                vfd = config.get("value_func", {})
                if not isinstance(vfd, dict) or "name" not in vfd or "params" not in vfd:
                    return fail(f"'value_func' must contain 'name' and 'params' in field '{field}'.")
        refresh_value_functions(store, settings)
        try:
            try:
                df = TimeSeriesSyntheticGenerator(schema).generate(n_rows=1)
            except Exception:
                df = AdvancedSyntheticDataGenerator(schema).generate(n_rows=1)
            if not isinstance(df, pd.DataFrame):
                return fail(f"Generated data is not a DataFrame for schema '{name}'.")
        except Exception:
            return fail("Provided schema is not valid.", 500)
        key = f"schemas/{name}.json"
        if store.exists(key):
            return fail(f"Schema name '{name}' already exists.")
        store.put(key, json.dumps(schema, indent=2).encode(), "application/json")
        return jsonify({"success": True, "message": f"Schema '{name}' saved successfully."}), 200

    @app.route('/get-data-schemas-api', methods=['GET'])
    def get_data_schemas():
        schema_name = request.args.get('name')
        if schema_name:
            try:
                schema = _load_schema(schema_name)
            except (LimitError, NotFound):
                return fail(f"Schema '{schema_name}' not found.", 404)
            return jsonify({"success": True, "schema_name": schema_name, "schema": schema}), 200
        names = [k[len("schemas/"):-len(".json")] for k in store.list("schemas/") if k.endswith(".json")]
        if not names:
            return fail("No data schemas available.", 404)
        return jsonify({"success": True, "schemas": names}), 200

    # this is the API for generating synthetic data
    @app.route('/generate-synthetic-data-api', methods=['POST'])
    def generate_synthetic_data():
        data = request.get_json(silent=True)
        if not data or "schema_name" not in data:
            return fail("Request must contain 'schema_name'.")
        schema_name = data["schema_name"]
        n_rows = data.get("n_rows", 10)
        try:
            schema = _load_schema(schema_name)
        except (LimitError, NotFound):
            return fail(f"Schema '{schema_name}' not found.", 404)
        try:
            check_request(schema, n_rows, settings.max_rows)
        except LimitError as e:
            return fail(str(e))
        refresh_value_functions(store, settings)
        if "timeseries_activity" in schema:
            generator, generator_type = TimeSeriesSyntheticGenerator(schema), "TimeSeriesSyntheticGenerator"
        else:
            generator, generator_type = AdvancedSyntheticDataGenerator(schema), "AdvancedSyntheticDataGenerator"
        try:
            generated_data = generator.generate(n_rows=n_rows)
            for col in generated_data.select_dtypes(include=['datetime64']).columns:
                generated_data[col] = generated_data[col].astype(str)
            generated_data = generated_data.replace({np.nan: None})
            stem = f"{int(time())}_{uuid.uuid4().hex[:8]}_{schema_name}_{generator_type}"
            html_key = f"{GENERATED_PREFIX}{stem}_plots.html"
            dataset_key = f"{GENERATED_PREFIX}{stem}_dataset.json"
            with tempfile.TemporaryDirectory() as tmp:
                SyntheticIO.audit_bias(generated_data, 10, True, tmp, "plots.html")
                store.put(html_key, (Path(tmp) / "plots.html").read_bytes(), "text/html")
            records = generated_data.to_dict(orient="records")
            store.put(dataset_key, json.dumps(records, indent=2, default=str).encode(), "application/json")
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            return fail(f"Error generating data: {e}", 500)
        return jsonify({"success": True, "schema_name": schema_name, "generator_type": generator_type,
                        "generated_data_path": dataset_key, "metadata": {"created_at": created_at},
                        "html_file_path": html_key}), 200

    def _generated(key, suffix):
        """Only generated outputs can be read back, by key, never by file path (was AUDIT S4)."""
        if not key:
            raise StorageError("No path provided.")
        check_key(key)
        if not key.startswith(GENERATED_PREFIX) or not key.endswith(suffix):
            raise StorageError(f"Only generated {suffix} outputs can be read.")
        return store.get(key)

    # this is the API for retrieving generated data, given the path returned by the generator
    @app.route('/get-data-api', methods=['GET'])
    def get_data():
        key = request.args.get('path')
        try:
            return jsonify({"success": True, "data": json.loads(_generated(key, ".json"))}), 200
        except NotFound:
            return fail(f"File '{key}' not found.", 404)
        except StorageError as e:
            return fail(str(e))

    # this is the API for retrieving a generated plot, given its path
    @app.route('/get-plot-api', methods=['GET'])
    def get_plot():
        key = request.args.get('path')
        try:
            html = _generated(key, ".html")
        except NotFound:
            return fail(f"Plot '{key}' not found.", 404)
        except StorageError as e:
            return fail(str(e))
        resp = Response(html, mimetype='text/html')
        # the page contains user-provided column names: run it in an opaque origin
        resp.headers["Content-Security-Policy"] = "sandbox allow-scripts"
        return resp

    return app


if __name__ == '__main__':
    _settings = settings_from_env()
    if _settings.legacy_code_exec:
        log.warning("DEBIAS_ENABLE_LEGACY_CODE_EXEC is on: /submit-value-function-api runs arbitrary code")
    create_app(_settings).run(host=_settings.host, port=_settings.port, debug=_settings.debug)
