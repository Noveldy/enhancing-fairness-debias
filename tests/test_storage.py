"""Phase 8: storage back ends and integrity checks of stored value functions (D5)."""
import json
import os
import uuid
from unittest import mock

import pytest

from service import value_store
from service.storage import LocalStore, MinioStore, NotFound, StorageError, check_key

SPEC = {"name": "lin", "doc": "Linear.", "params": {"a": 2.0}, "expression": "a * i"}


@pytest.mark.parametrize("key", ["../x", "/etc/passwd", "a/../b", "a//b", ".hidden", "a/.b", "a\\b", "", "a/",
                                 "x" * 600, "a b", "%2e%2e/x"])
def test_bad_keys(key):
    with pytest.raises(StorageError):
        check_key(key)


def test_local_store_roundtrip_and_symlink_escape(tmp_path):
    s = LocalStore(tmp_path / "root")
    s.put("schemas/a.json", b"{}")
    assert s.get("schemas/a.json") == b"{}" and s.exists("schemas/a.json")
    assert s.list("schemas/") == ["schemas/a.json"]
    with pytest.raises(NotFound):
        s.get("schemas/b.json")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_text("x")
    os.symlink(tmp_path / "outside", tmp_path / "root" / "link")
    with pytest.raises(StorageError):
        s.get("link/secret")
    s.delete("schemas/a.json")
    assert not s.exists("schemas/a.json")


def test_integrity_sha256(tmp_path):
    s = LocalStore(tmp_path)
    value_store.save_spec(s, SPEC)
    good, bad = value_store.load_specs(s)
    assert list(good) == ["lin"] and good["lin"]({})(3) == 6.0 and not bad
    env = json.loads(s.get("value_functions/lin.json"))
    env["spec"]["expression"] = "a * i + 1000"                 # tampered, hash not updated
    s.put("value_functions/lin.json", json.dumps(env).encode())
    good, bad = value_store.load_specs(s)
    assert not good and "sha256" in bad[0][1]


def test_integrity_hmac(tmp_path):
    s = LocalStore(tmp_path)
    value_store.save_spec(s, SPEC, hmac_key="k1")
    assert list(value_store.load_specs(s, "k1")[0]) == ["lin"]
    assert not value_store.load_specs(s, "other-key")[0]       # wrong key
    env = json.loads(s.get("value_functions/lin.json"))
    env["spec"]["expression"] = "a * i + 1000"
    forged = value_store.seal({k: v for k, v in env.items() if k not in ("sha256", "hmac_sha256")}, "spec")
    s.put("value_functions/lin.json", forged)                  # attacker recomputed sha256, has no key
    good, bad = value_store.load_specs(s, "k1")
    assert not good and "HMAC" in bad[0][1]


def test_planted_code_in_store_is_not_run(tmp_path):
    s = LocalStore(tmp_path)
    sentinel = tmp_path / "pwned"
    planted = {"format": 1, "kind": "declarative",
               "spec": {"name": "x", "doc": "d", "params": {},
                        "expression": f"__import__('os').system('touch {sentinel}')"}}
    s.put("value_functions/x.json", value_store.seal(planted, "spec"))
    s.put("value_functions/y.json", b"\x80\x04 pickle-like bytes")
    legacy = {"format": 1, "kind": "legacy_code", "code": f"open({str(sentinel)!r},'w')"}
    s.put("value_functions/z.json", value_store.seal(legacy, "code"))
    good, bad = value_store.load_specs(s)
    assert not good and len(bad) == 3 and not sentinel.exists()


def _fake_minio():
    """In-memory stand-in for minio.Minio with the calls MinioStore uses."""
    from minio.error import S3Error
    objects = {}
    client = mock.MagicMock()
    client.bucket_exists.return_value = False

    def put_object(bucket, name, data, length, content_type):
        objects[name] = data.read(length)

    def missing(name):
        return S3Error(mock.MagicMock(), "NoSuchKey", "missing", name, "r", "h")

    def get_object(bucket, name):
        if name not in objects:
            raise missing(name)
        resp = mock.MagicMock()
        resp.read.return_value = objects[name]
        return resp

    def stat_object(bucket, name):
        if name not in objects:
            raise missing(name)

    client.put_object.side_effect = put_object
    client.get_object.side_effect = get_object
    client.stat_object.side_effect = stat_object
    client.remove_object.side_effect = lambda b, n: objects.pop(n, None)
    client.list_objects.side_effect = lambda b, prefix, recursive: [
        mock.MagicMock(object_name=k) for k in sorted(objects) if k.startswith(prefix)]
    return client, objects


def test_minio_store_with_mocked_client():
    client, objects = _fake_minio()
    s = MinioStore(client, "debias", prefix="tenant1")
    client.make_bucket.assert_called_once_with("debias")
    value_store.save_spec(s, SPEC)
    assert list(objects) == ["tenant1/value_functions/lin.json"]
    assert s.list("value_functions/") == ["value_functions/lin.json"]
    assert value_store.load_specs(s)[0]["lin"]({})(1) == 2.0
    with pytest.raises(NotFound):
        s.get("value_functions/none.json")
    with pytest.raises(StorageError):
        s.put("../escape", b"x")
    s.delete("value_functions/lin.json")
    assert not s.exists("value_functions/lin.json")


def test_store_from_env_selects_backend(tmp_path):
    from service.storage import store_from_env
    assert store_from_env({"DEBIAS_STORAGE_DIR": str(tmp_path)}).kind == "local"
    with mock.patch("minio.Minio") as m:
        m.return_value.bucket_exists.return_value = True
        s = store_from_env({"DEBIAS_MINIO_ENDPOINT": "localhost:9000", "DEBIAS_MINIO_SECURE": "0",
                            "DEBIAS_MINIO_ACCESS_KEY": "a", "DEBIAS_MINIO_SECRET_KEY": "b"})
    assert s.kind == "minio"
    m.assert_called_once_with("localhost:9000", access_key="a", secret_key="b", secure=False)


@pytest.mark.skipif(not os.environ.get("DEBIAS_TEST_MINIO_ENDPOINT"),
                    reason="set DEBIAS_TEST_MINIO_ENDPOINT (+ _ACCESS_KEY, _SECRET_KEY) to test a real MinIO")
def test_real_minio_roundtrip():
    from service.storage import minio_from_env
    env = {"DEBIAS_MINIO_ENDPOINT": os.environ["DEBIAS_TEST_MINIO_ENDPOINT"],
           "DEBIAS_MINIO_ACCESS_KEY": os.environ.get("DEBIAS_TEST_MINIO_ACCESS_KEY", "minioadmin"),
           "DEBIAS_MINIO_SECRET_KEY": os.environ.get("DEBIAS_TEST_MINIO_SECRET_KEY", "minioadmin"),
           "DEBIAS_MINIO_SECURE": "0", "DEBIAS_MINIO_BUCKET": "debias-test",
           "DEBIAS_MINIO_PREFIX": f"t{uuid.uuid4().hex[:8]}"}
    s = minio_from_env(env)
    value_store.save_spec(s, SPEC, hmac_key="k")
    assert value_store.load_specs(s, "k")[0]["lin"]({})(2) == 4.0
    s.put("generated/x.json", b"[]")
    assert sorted(s.list()) == ["generated/x.json", "value_functions/lin.json"]
    for k in s.list():
        s.delete(k)
    assert s.list() == []
