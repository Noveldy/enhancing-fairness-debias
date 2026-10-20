"""MinIO connector for the DEBIAS API storage (Phase 8, author decision D5).

Settings come from the same environment variables the API uses (service/storage.py):
    DEBIAS_MINIO_ENDPOINT=localhost:9000 DEBIAS_MINIO_ACCESS_KEY=... DEBIAS_MINIO_SECRET_KEY=... \
    DEBIAS_MINIO_SECURE=0 venv/bin/python scripts/minio_connector.py check

Commands:
    check                      connect, create the bucket if needed, write/read/delete a probe object
    ls [PREFIX]                list keys
    put FILE KEY               upload a local file
    get KEY FILE               download to a local file
    sync-local [--from DIR] [--overwrite]
                               copy everything from the local store (default <repo>/storage) into MinIO
    verify                     check the integrity of every stored value function (sha256 / HMAC)
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from service import value_store  # noqa: E402
from service.storage import LocalStore, minio_from_env  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("ls")
    p.add_argument("prefix", nargs="?", default="")
    p = sub.add_parser("put")
    p.add_argument("file")
    p.add_argument("key")
    p = sub.add_parser("get")
    p.add_argument("key")
    p.add_argument("file")
    p = sub.add_parser("sync-local")
    p.add_argument("--from", dest="src", default=os.environ.get("DEBIAS_STORAGE_DIR") or REPO_ROOT / "storage")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("verify")
    args = ap.parse_args(argv)

    if not os.environ.get("DEBIAS_MINIO_ENDPOINT"):
        ap.error("set DEBIAS_MINIO_ENDPOINT (and the access/secret keys) first")
    store = minio_from_env()

    if args.cmd == "check":
        key = f"connector-probe/{uuid.uuid4().hex}.txt"
        store.put(key, b"ok", "text/plain")
        assert store.get(key) == b"ok" and store.exists(key)
        store.delete(key)
        assert not store.exists(key)
        print(f"OK: {store.describe()} (write, read, delete)")
    elif args.cmd == "ls":
        for k in store.list(args.prefix):
            print(k)
    elif args.cmd == "put":
        ctype = mimetypes.guess_type(args.file)[0] or "application/octet-stream"
        store.put(args.key, Path(args.file).read_bytes(), ctype)
        print(f"uploaded {args.file} -> {args.key}")
    elif args.cmd == "get":
        Path(args.file).write_bytes(store.get(args.key))
        print(f"downloaded {args.key} -> {args.file}")
    elif args.cmd == "sync-local":
        src = LocalStore(args.src)
        copied = skipped = 0
        for k in src.list():
            if store.exists(k) and not args.overwrite:
                skipped += 1
                continue
            store.put(k, src.get(k), mimetypes.guess_type(k)[0] or "application/octet-stream")
            copied += 1
        print(f"copied {copied}, skipped {skipped} existing (from {src.describe()} to {store.describe()})")
    elif args.cmd == "verify":
        good, bad = value_store.load_specs(store, os.environ.get("DEBIAS_STORE_HMAC_KEY") or None)
        print(f"{len(good)} value functions pass the integrity checks")
        for key, reason in bad:
            print(f"REJECTED {key}: {reason}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
