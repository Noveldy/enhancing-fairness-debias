"""Stored value functions with integrity checks (Phase 8).

Each user-defined value function is stored as JSON data under `value_functions/<name>.json`:

    {"format": 1, "kind": "declarative", "spec": {...}, "sha256": "<hex>", "hmac_sha256": "<hex>|null"}

`sha256` is over the canonical JSON of `spec` and catches corruption or a hand edit that forgets to update
it. If DEBIAS_STORE_HMAC_KEY is set, an HMAC-SHA256 with that server-side key is added and *required* on
load, so someone who can write to the store but does not have the key cannot plant or alter a function.
A spec is re-validated on every load and only ever interpreted by service.expressions; nothing loaded from
the store is passed to eval, exec, compile, pickle or import.

Legacy code submissions (only when DEBIAS_ENABLE_LEGACY_CODE_EXEC=1) are stored separately under
`legacy_value_functions/<name>.json` with the same checks, and are loaded only while that flag is set.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging

from service.expressions import ExpressionError, make_factory, validate_spec
from service.storage import NotFound

log = logging.getLogger(__name__)
PREFIX = "value_functions/"
LEGACY_PREFIX = "legacy_value_functions/"


class IntegrityError(Exception):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digests(payload: bytes, hmac_key):
    sha = hashlib.sha256(payload).hexdigest()
    mac = hmac.new(hmac_key.encode(), payload, hashlib.sha256).hexdigest() if hmac_key else None
    return sha, mac


def seal(body: dict, payload_field: str, hmac_key=None) -> bytes:
    sha, mac = _digests(canonical(body[payload_field]), hmac_key)
    # hashes use the sorted canonical form; the stored file keeps the key order (parameter order matters)
    return json.dumps({**body, "sha256": sha, "hmac_sha256": mac}, indent=1, ensure_ascii=False).encode()


def unseal(raw: bytes, payload_field: str, hmac_key=None) -> dict:
    try:
        env = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise IntegrityError("not valid JSON") from None
    if not isinstance(env, dict) or env.get("format") != 1 or payload_field not in env:
        raise IntegrityError("unknown envelope format")
    sha, mac = _digests(canonical(env[payload_field]), hmac_key)
    if not hmac.compare_digest(str(env.get("sha256")), sha):
        raise IntegrityError("sha256 mismatch: stored content was changed")
    if hmac_key:
        if not env.get("hmac_sha256") or not hmac.compare_digest(str(env["hmac_sha256"]), mac):
            raise IntegrityError("HMAC missing or wrong: not written by this server")
    return env


def save_spec(store, spec, hmac_key=None, reserved=()):
    spec = validate_spec(spec)
    key = PREFIX + spec["name"] + ".json"
    if spec["name"] in reserved or store.exists(key):
        raise ExpressionError(f"value function '{spec['name']}' already exists")
    make_factory(spec)                                          # compiles: fails here, not at load time
    store.put(key, seal({"format": 1, "kind": "declarative", "spec": spec}, "spec", hmac_key),
              "application/json")
    return spec


def load_specs(store, hmac_key=None):
    """{name: factory} for every stored spec that passes the checks, and [(key, reason)] for the rest."""
    good, bad = {}, []
    for key in store.list(PREFIX):
        try:
            env = unseal(store.get(key), "spec", hmac_key)
            if env.get("kind") != "declarative":
                raise IntegrityError("not a declarative value function")
            spec = validate_spec(env["spec"])
            if key != PREFIX + spec["name"] + ".json":
                raise IntegrityError("file name does not match the function name")
            good[spec["name"]] = make_factory(spec)
        except (IntegrityError, ExpressionError, NotFound) as e:
            bad.append((key, str(e)))
            log.warning("rejected stored value function %s: %s", key, e)
    return good, bad


# -- legacy code path (arbitrary code; only with DEBIAS_ENABLE_LEGACY_CODE_EXEC=1) ------------------------
def save_legacy_code(store, name, code, hmac_key=None):
    key = LEGACY_PREFIX + name + ".json"
    if store.exists(key):
        raise ExpressionError(f"value function '{name}' already exists")
    store.put(key, seal({"format": 1, "kind": "legacy_code", "name": name, "code": code}, "code", hmac_key),
              "application/json")


def load_legacy_code(store, namespace_factory, hmac_key=None):
    """Run stored legacy code (integrity-checked) and collect its factories. ARBITRARY CODE EXECUTION."""
    good, bad = {}, []
    for key in store.list(LEGACY_PREFIX):
        try:
            env = unseal(store.get(key), "code", hmac_key)
            ns = namespace_factory()
            exec(env["code"], ns)                               # noqa: S102  legacy, opt-in only
            good.update({k: v for k, v in ns["__registered__"].items()})
        except Exception as e:                                   # stored code may raise anything
            bad.append((key, f"{e.__class__.__name__}: {e}"))
            log.warning("rejected legacy value function %s: %s", key, e)
    return good, bad
