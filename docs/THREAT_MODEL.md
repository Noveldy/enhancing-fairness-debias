# Threat model: DEBIAS back end (Phase 8)

Answers Reviewer 2's comment on the security of user-defined functions. The audit findings are in
`docs/AUDIT.md` §9 (S1–S9). Author decisions:
- **D4 (a):** replace code with declarative, allow-listed value functions. The old code path stays behind a flag that is off by default. The Flutter UI (`../debias`) is out of scope.
- **D5:** add a MinIO storage connector.

## 1. System

| Component | What it is | Trust |
|---|---|---|
| `apis.py` | Flask REST API: bias check of an uploaded dataset, value functions, data schemas, synthetic data, plots | the system under analysis |
| Flutter UI (`../debias`) | client on desktop, mobile or web; calls the API over HTTP | untrusted input source (any client can call the API directly) |
| Storage (`service/storage.py`) | local directory (default `storage/`) or a MinIO bucket | holds uploads, schemas, value functions, generated data |
| Pipeline (`main.py`, `bias_identification/`, `bias_mitigation/`) | runs on uploaded data; loads a local BERT checkpoint (safetensors) | code shipped with the repository |
| Built-in value functions (`bias_mitigation/synthetic_data/value_functions.py`) | 16 formulas shipped with the code | trusted |

## 2. Assets

1. **The server host.** Code execution on it exposes everything below, plus credentials, SSH keys, other projects and the local network.
2. **Uploaded datasets.** These contain personal data by design: the tool looks for sensitive attributes such as sex, race and age.
3. **Stored value functions and schemas.** They are shared by all users, so changing one changes everyone's synthetic data.
4. **Generated datasets and reports.**
5. **Secrets:** MinIO access and secret keys, the HMAC key and the API key.
6. **Availability** of the service and the host: CPU, memory and disk.

## 3. Attackers

| ID | Attacker | Capability |
|---|---|---|
| A1 | network attacker | can send HTTP requests to the API. The original bound to `0.0.0.0` with no authentication, so this meant anyone on the LAN or the internet. |
| A2 | malicious user of a shared deployment | uses the API normally but sends crafted inputs |
| A3 | malicious web page opened by a user | makes the user's browser call an API on `localhost` (CORS was open to any origin) |
| A4 | someone with write access to the storage (MinIO bucket or storage directory) but not to the server | plants or alters stored functions |
| A5 | supply chain (PyPI packages, model files) | out of scope here; versions are pinned (`requirements-*.txt`) |

## 4. Entry points and what the original code did with them

| Route / input | Original behaviour (AUDIT) | Impact |
|---|---|---|
| `POST /submit-value-function-api` `code` | `exec(code)` with full builtins **before** any check (S1). Then the code was appended to `value_functions.py` (S2), which is imported on every start. | **Remote code execution** (A1–A3): read or write any file the process can reach (datasets, keys); network access for exfiltration and pivoting; fork bombs, memory bombs and infinite loops; **persistence**, because the code reruns at every server start for every user, with no review and no integrity check. |
| `POST /data-bias-api` `detect_outliers` | `eval()` of a form field (S3) | remote code execution |
| `GET /get-data-api?path=`, `/get-plot-api?path=` | opened any path (S4) | read any file the process can read |
| `POST /submit-data-schema-api` `name`, `POST /generate-synthetic-data-api` `schema_name` | interpolated into file paths (S5) | path traversal on write |
| `app.run(debug=True, host="0.0.0.0")` | Werkzeug debugger reachable from the network (S6) | remote code execution through the debugger console |
| all routes | no authentication; CORS allowed any origin (S7) | A1 and A3 can use everything above |
| schema fields `n_rows`, `periods`, `length`, `pattern` | unbounded | memory or CPU exhaustion (for example `pattern: "a{999999999}"`) |
| uploads | `secure_filename`, but one shared folder (S9) | users overwrite each other's files |

A "restricted" `exec`, such as removing builtins, would not be enough. Python objects reachable from numpy or from any literal lead back to `__import__` (for example `().__class__.__mro__[1].__subclasses__()`). That is why the plan prefers removing code execution to sandboxing it.

## 5. Mitigations implemented

| # | Mitigation | Where | Test (`tests/test_security.py`, `tests/test_storage.py`) |
|---|---|---|---|
| S1 | **No code execution.** A value function is data: `{name, doc, params, expression}`. The expression is parsed with `ast.parse(mode="eval")` and checked node by node against an allow-list: numbers, `i`, declared parameters, arithmetic, comparisons, `a if c else b`, and 91 allow-listed function names: math functions, `min`, `max`, `clip` and random draws, each counted once per spelling (`np.`, `numpy.`, `math.`). Attribute access is allowed only as a complete dotted name from that list, looked up in our own table. Evaluation uses our own tree interpreter; nothing reaches `eval`, `exec`, `compile` or `getattr`. | `service/expressions.py` | `test_expression_attacks_rejected`, `test_api_rejects_attacks_in_ui_code_and_in_specs` (os.system, file read and write, urllib, socket, `__class__` escapes, `getattr`, lambda, comprehension, infinite iterator, `size=` memory bomb, string bomb, walrus, f-string) |
| S1 | **The UI keeps working without exec.** The UI posts a Python template (S8). The server reads its syntax tree without running it: one decorated factory, `p = params.get("p", <number>)` lines, `return lambda i: <expression>`, an optional `_params` list. Anything else is rejected. All 16 built-in functions fit this form and give identical outputs. | `spec_from_ui_template` | `test_original_audit_exploit_does_not_run` (code before the function, a payload hidden in a parameter line, `while True`), `test_ui_template_accepted_and_evaluated_without_exec`, `test_every_builtin_value_function_is_expressible` |
| S1 | **Resource limits in expressions:** at most 2000 characters, 300 nodes and nesting depth 40; fixed call arity (no `size=`, so one call returns one number); float64 arithmetic with errors silenced (`9**9**9**9` gives `inf` in microseconds); integer literals that overflow a float are rejected. The language has no loops, so evaluation time is bounded by the node count. | `service/expressions.py` | `test_cpu_and_overflow_bounded` |
| S2 | **No writes into the source tree.** Functions are stored as JSON under `value_functions/<name>.json` in the storage, never appended to a `.py` file. On load each one is re-validated and only interpreted, never executed. | `service/value_store.py` | `test_planted_code_in_store_is_not_run` |
| S2 | **Integrity checks:** SHA-256 over the canonical spec, plus HMAC-SHA256 with a server key (`DEBIAS_STORE_HMAC_KEY`). When the key is set, the HMAC is required, so A4 cannot plant or alter a function without it. Functions that fail are skipped and logged. `scripts/minio_connector.py verify` reports them. | `service/value_store.py` | `test_integrity_sha256`, `test_integrity_hmac` |
| — | **Legacy endpoint (D4):** the original `exec` path runs only with `DEBIAS_ENABLE_LEGACY_CODE_EXEC=1` (off by default) and only for code that does not fit the safe template. **Enabling it restores arbitrary code execution.** Submissions are stored with the same integrity envelope under `legacy_value_functions/`, outside the source tree, and are loaded only while the flag is on. | `apis.py`, `service/value_store.py` | `test_legacy_exec_disabled_by_default_and_opt_in` |
| S3 | `detect_outliers` is parsed as a boolean (`true` or `false`) | `apis.py` | `test_detect_outliers_not_evaluated` |
| S4 | No client-supplied paths. `/get-data-api` and `/get-plot-api` accept only storage keys under `generated/` with the right suffix. Keys are checked against a strict pattern: no `..`, no leading `/` or `.`, no `\`. In the local store, the resolved path must stay inside the root, which blocks symlink escapes. | `service/storage.py` | `test_no_arbitrary_file_read`, `test_bad_keys`, `test_local_store_roundtrip_and_symlink_escape` |
| S5 | Schema names must match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` | `service/limits.py` | `test_schema_name_traversal_rejected` |
| S6 | Binds to `127.0.0.1` with the debugger off by default (`DEBIAS_HOST`, `DEBIAS_DEBUG`) | `service/config.py` | `test_safe_defaults`; manual check: `ss` shows `127.0.0.1:5077`, and `/console` returns 404 |
| S7 | Optional API key (`DEBIAS_API_KEY`, `X-API-Key` header, constant-time comparison). CORS is limited to `http(s)://localhost` and `127.0.0.1` origins (`DEBIAS_CORS_ORIGINS`). | `apis.py`, `service/config.py` | `test_api_key_and_cors` |
| — | **Size limits:** produced rows `n_rows × Σ periods` ≤ `DEBIAS_MAX_ROWS` (default 1,000,000); `string` length ≤ 10,000; `pattern` ≤ 500 characters and ≤ 10,000 characters of output (`*` counted as rstr's 100); ≤ 500 columns; ≤ 10,000 categories; uploads ≤ `DEBIAS_MAX_UPLOAD_MB` (default 200) | `service/limits.py`, `apis.py` | `test_memory_bombs_rejected`, `test_upload_size_limit` |
| S9 | Each upload is stored under its own random key (`uploads/<uuid>_<name>`) | `apis.py` | — |
| — | Generated HTML reports contain user-chosen column names, so they are served with `Content-Security-Policy: sandbox allow-scripts`, which runs them in an opaque origin | `apis.py` | `test_generated_plot_is_sandboxed` |
| D5 | **Storage:** a local directory or MinIO (`DEBIAS_MINIO_*`, HTTPS by default). `scripts/minio_connector.py` can `check`, `ls`, `put`, `get`, `sync-local` (migrate the local store) and `verify` | `service/storage.py`, `scripts/minio_connector.py` | `test_minio_store_with_mocked_client`, `test_store_from_env_selects_backend`, `test_real_minio_roundtrip` (needs a server; see §8) |

## 6. Residual risks (not solved here)

- **No user accounts.** The paper says stored files are "only available to the user that requested them" (§2.5.2). This is still **not true**: there is one optional shared API key and no per-user storage. Generated outputs have hard-to-guess keys (a timestamp plus 32 random bits), but that is not access control.
- **The current Flutter UI does not send `X-API-Key`.** Turning on API-key mode therefore breaks the UI until the UI is updated, which is out of scope.
- **CPU cost of `/data-bias-api`.** A 200 MB upload can occupy the server for a long time: the pipeline has no per-request time limit and no process isolation. For a shared deployment, run the API under a WSGI server with worker timeouts, request rate limits and TLS. The Flask development server is for local use only.
- **Synthetic generation is bounded but not instant.** One million rows with a 300-node expression is Python-level work (seconds to minutes).
- **Without `DEBIAS_STORE_HMAC_KEY`**, the SHA-256 detects accidental or naive changes but not a deliberate attacker with write access to the store. Set the key whenever the bucket or directory is shared.
- **The legacy flag** re-enables remote code execution. Integrity checks on stored legacy code only prove who wrote it, not that it is safe.
- **Data retention:** uploads and generated data are kept until deleted by hand. There is no retention policy for the personal data in uploads.
- **Model files** are loaded from the local repository (`model.safetensors`; the pickled `training_args.bin` is not loaded). Replacing them requires write access to the server.
- **The UI still builds Python source by string interpolation** (S8). The server no longer runs it, but the UI and API contract should move to the JSON form `{name, doc, params, expression}`, which the endpoint already accepts.

## 7. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DEBIAS_HOST`, `DEBIAS_PORT` | `127.0.0.1`, `5000` | bind address |
| `DEBIAS_DEBUG` | `0` | Werkzeug debugger. Never enable on a reachable host. |
| `DEBIAS_API_KEY` | unset | if set, every request needs `X-API-Key` |
| `DEBIAS_CORS_ORIGINS` | localhost origins | comma-separated origins (regexes allowed) |
| `DEBIAS_STORE_HMAC_KEY` | unset | signs stored value functions; required on load when set |
| `DEBIAS_ENABLE_LEGACY_CODE_EXEC` | `0` | **restores arbitrary code execution** for the old code path |
| `DEBIAS_MAX_ROWS`, `DEBIAS_MAX_UPLOAD_MB` | `1000000`, `200` | size limits |
| `DEBIAS_STORAGE_DIR` | `<repo>/storage` | local storage root (git-ignored) |
| `DEBIAS_MINIO_ENDPOINT`, `_ACCESS_KEY`, `_SECRET_KEY`, `_BUCKET`, `_PREFIX`, `_SECURE` | unset, –, –, `debias`, –, `1` | MinIO; it is used when the endpoint is set |

## 8. How this was tested (2026-10-20)

- `venv/bin/python -m pytest tests/test_security.py tests/test_storage.py`. Each attack writes a sentinel file or reads `/etc/passwd` if it succeeds; the tests check both the HTTP status and that the side effect did not happen.
- **Real object-store round trip.** Docker's daemon is not running on this machine, and `dl.min.io` could not be reached (TLS connection closed). The S3 tests therefore ran against moto's S3-compatible server (moto 5.2.4, in a throwaway virtual environment, loopback only), using the real `minio` client over HTTP:
  - `DEBIAS_TEST_MINIO_ENDPOINT=127.0.0.1:9123 DEBIAS_TEST_MINIO_ACCESS_KEY=testing DEBIAS_TEST_MINIO_SECRET_KEY=testing pytest tests/test_storage.py -k real` passed;
  - the connector's `check`, `sync-local` (2 copied, then 2 skipped), `ls`, `get`, `put` and `verify` all worked, and `verify` rejected a tampered function with exit code 1;
  - the full API flow (submit function → schema → generate → read data) ran with MinIO storage.

  **A run against a real MinIO server is still open.** To run it:

  ```bash
  docker run -p 9000:9000 minio/minio server /data
  DEBIAS_TEST_MINIO_ENDPOINT=127.0.0.1:9000 venv/bin/python -m pytest tests/test_storage.py -k real
  ```
- **Live server.** `python apis.py` listened on `127.0.0.1` only; `/get-data-api?path=/etc/passwd` returned 400; `/console` returned 404.
