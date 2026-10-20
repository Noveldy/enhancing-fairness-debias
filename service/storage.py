"""Object storage for the API: local directory (default) or MinIO (Phase 8, author decision D5).

The paper (Section 2.5.2) says datasets, data schemas and value functions are kept in MinIO; the original
code wrote them into the source tree instead (AUDIT S2). Both back ends here store opaque bytes under
validated keys such as `schemas/adult_like.json`:

    store = store_from_env()          # MinIO if DEBIAS_MINIO_ENDPOINT is set, else a local directory
    store.put("schemas/x.json", b"{...}", "application/json")
    store.get("schemas/x.json")

Environment:
    DEBIAS_STORAGE_DIR       local root (default: <repo>/storage)
    DEBIAS_MINIO_ENDPOINT    host:port of a MinIO / S3 server; enables MinIO
    DEBIAS_MINIO_ACCESS_KEY, DEBIAS_MINIO_SECRET_KEY
    DEBIAS_MINIO_BUCKET      default "debias"
    DEBIAS_MINIO_PREFIX      optional key prefix inside the bucket
    DEBIAS_MINIO_SECURE      "1" (default) for HTTPS, "0" for plain HTTP (local testing only)
"""
from __future__ import annotations

import io
import os
import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(/[A-Za-z0-9][A-Za-z0-9._-]{0,127}){0,7}$")
MAX_KEY_CHARS = 512


class StorageError(Exception):
    pass


class NotFound(StorageError):
    pass


def check_key(key: str) -> str:
    """Keys are relative, '/'-separated, each part starts with a letter or digit: no '..', no '/x', no '\\'."""
    if not isinstance(key, str) or len(key) > MAX_KEY_CHARS or not KEY_RE.match(key):
        raise StorageError(f"invalid storage key: {key!r}")
    return key


class LocalStore:
    kind = "local"

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        p = (self.root / check_key(key)).resolve()
        if not p.is_relative_to(self.root):          # e.g. a symlink inside the root pointing out of it
            raise StorageError(f"key escapes the storage root: {key!r}")
        return p

    def put(self, key, data: bytes, content_type="application/octet-stream"):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)                        # atomic: readers never see half a file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, key) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise NotFound(key)
        return p.read_bytes()

    def exists(self, key) -> bool:
        return self._path(key).is_file()

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix=""):
        base = self.root / prefix if prefix else self.root
        if prefix:
            check_key(prefix.rstrip("/"))
        if not base.exists():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file() and not p.name.startswith(".tmp-"):
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)

    def describe(self):
        return f"local directory {self.root}"


class MinioStore:
    kind = "minio"

    def __init__(self, client, bucket, prefix=""):
        self.client, self.bucket = client, bucket
        self.prefix = (check_key(prefix.strip("/")) + "/") if prefix.strip("/") else ""
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

    def _name(self, key):
        return self.prefix + check_key(key)

    def put(self, key, data: bytes, content_type="application/octet-stream"):
        self.client.put_object(self.bucket, self._name(key), io.BytesIO(data), length=len(data),
                               content_type=content_type)

    def get(self, key) -> bytes:
        from minio.error import S3Error
        try:
            resp = self.client.get_object(self.bucket, self._name(key))
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject"):
                raise NotFound(key) from None
            raise
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key) -> bool:
        from minio.error import S3Error
        try:
            self.client.stat_object(self.bucket, self._name(key))
            return True
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject", "NotFound"):
                return False
            raise

    def delete(self, key):
        self.client.remove_object(self.bucket, self._name(key))

    def list(self, prefix=""):
        if prefix:
            check_key(prefix.rstrip("/"))
        objs = self.client.list_objects(self.bucket, prefix=self.prefix + prefix, recursive=True)
        return sorted(o.object_name[len(self.prefix):] for o in objs)

    def describe(self):
        return f"MinIO bucket '{self.bucket}' at {self.client._base_url.host}" + (
            f" (prefix {self.prefix})" if self.prefix else "")


def minio_from_env(env=os.environ):
    from minio import Minio
    endpoint = env["DEBIAS_MINIO_ENDPOINT"]
    client = Minio(endpoint, access_key=env.get("DEBIAS_MINIO_ACCESS_KEY"),
                   secret_key=env.get("DEBIAS_MINIO_SECRET_KEY"),
                   secure=env.get("DEBIAS_MINIO_SECURE", "1") != "0")
    return MinioStore(client, env.get("DEBIAS_MINIO_BUCKET", "debias"), env.get("DEBIAS_MINIO_PREFIX", ""))


def store_from_env(env=os.environ):
    if env.get("DEBIAS_MINIO_ENDPOINT"):
        return minio_from_env(env)
    return LocalStore(env.get("DEBIAS_STORAGE_DIR") or REPO_ROOT / "storage")
