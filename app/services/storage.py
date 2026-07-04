"""Raw-file storage: S3 when configured, local disk otherwise.

The local fallback keeps development and CI working before the AWS bucket is
provisioned (documented in phase-1 completion report §8). The stored key format
is identical either way, so switching to S3 is a pure config change.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import boto3

from app.core.config import get_settings

LOCAL_ROOT = Path("var/uploads")


def make_key(file_name: str) -> str:
    today = datetime.now(UTC).strftime("%Y/%m/%d")
    return f"uploads/{today}/{uuid4().hex}-{Path(file_name).name}"


class FileStorage:
    def __init__(self) -> None:
        settings = get_settings()
        self._bucket = settings.s3_bucket
        self._use_s3 = bool(settings.supabase_url and settings.s3_bucket and settings.env != "dev")
        # explicit opt-in beats guessing: S3 only when AWS creds resolve
        try:
            self._use_s3 = self._use_s3 and boto3.Session().get_credentials() is not None
        except Exception:
            self._use_s3 = False

    def save(self, key: str, data: bytes) -> str:
        if self._use_s3:
            boto3.client("s3").put_object(Bucket=self._bucket, Key=key, Body=data)
            return f"s3://{self._bucket}/{key}"
        path = LOCAL_ROOT / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def load(self, key: str) -> bytes:
        if self._use_s3:
            resp = boto3.client("s3").get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        return (LOCAL_ROOT / key).read_bytes()
