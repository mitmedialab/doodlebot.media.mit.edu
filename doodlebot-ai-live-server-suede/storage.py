"""S3-backed blob storage for the DoodleBot v2 routes.

Replaces the RESOURCES_DIR / COMBINED_DIR disk trees with an S3 bucket. All
methods are *synchronous* (boto3 is a blocking HTTP client); call them from
async code via ``await asyncio.to_thread(...)`` so the event loop stays free.

Configuration (all via environment):

  S3_BUCKET             required — the bucket name
  AWS_REGION            e.g. "us-west-2"
  AWS_ACCESS_KEY_ID /   standard boto3 credential chain; on EC2/ECS prefer an
  AWS_SECRET_ACCESS_KEY IAM role and set neither
  S3_ENDPOINT_URL       optional — point at MinIO/LocalStack for local dev

Key layout in the bucket:

  resources/{sha256}{ext}     served blobs (sketch PNGs, vectorization SVGs)
  combined/combined_{ts}.png  debug snapshots of GPT Image 1 output
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Mirrors the mapping in the routes module so ids stay interchangeable.
_EXT_BY_CT = {"image/png": ".png", "image/svg+xml": ".svg", "image/jpeg": ".jpg"}
_CT_BY_EXT = {ext: ct for ct, ext in _EXT_BY_CT.items()}

RESOURCE_PREFIX = "resources/"
COMBINED_PREFIX = "combined/"

_IMMUTABLE = "public, max-age=31536000, immutable"


@dataclass(frozen=True)
class StoredResource:
    """A servable blob living in S3 (replaces the disk-backed Resource)."""

    key: str
    content_type: str


class S3Storage:
    def __init__(
        self,
        bucket: str | None = None,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self._s3 = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_REGION"),
            endpoint_url=endpoint_url or os.environ.get("S3_ENDPOINT_URL") or None,
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )

    # -- writes ---------------------------------------------------------

    def put_resource(self, body: bytes, content_type: str) -> str:
        """Content-address ``body`` and upload it (idempotently). Returns the
        resource id (sha256 hex) — the same id scheme the disk version used.

        Skips the upload if the object already exists, so re-submitting an
        identical sketch costs one HEAD request instead of a re-upload."""
        resource_id = hashlib.sha256(body).hexdigest()
        key = self.resource_key(resource_id, content_type)
        if not self._exists(key):
            self._s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl=_IMMUTABLE,
            )
        return resource_id

    def put_resource_at(self, resource_id: str, body: bytes, content_type: str) -> None:
        """Upload ``body`` under a *caller-chosen* resource id rather than a
        content hash. Used for admin-gated vectorizations, whose id is a stable
        locator minted before the final bytes exist (the admin may hand back a
        simplified command set), so the served blob is filled in at approval time
        under an id the admin already holds. Overwrites any existing object."""
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self.resource_key(resource_id, content_type),
            Body=body,
            ContentType=content_type,
            CacheControl=_IMMUTABLE,
        )

    def put_combined_debug(self, png_bytes: bytes, name: str) -> None:
        """Fire-and-forget archive of a combined PNG (replaces COMBINED_DIR)."""
        self._s3.put_object(
            Bucket=self.bucket,
            Key=f"{COMBINED_PREFIX}{name}",
            Body=png_bytes,
            ContentType="image/png",
        )

    # -- reads ----------------------------------------------------------

    def read(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Time-limited direct-download URL. Computed locally (no network),
        so it's safe to call on the event loop. Max ExpiresIn is 7 days."""
        return self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )

    # -- startup recovery -------------------------------------------------

    def list_resources(self) -> dict[str, StoredResource]:
        """Enumerate every served blob under resources/, keyed by resource id.
        Drop-in replacement for the RESOURCES_DIR.iterdir() startup scan."""
        found: dict[str, StoredResource] = {}
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=RESOURCE_PREFIX):
            for item in page.get("Contents", []):
                key: str = item["Key"]
                stem, dot, ext = key[len(RESOURCE_PREFIX) :].rpartition(".")
                content_type = _CT_BY_EXT.get(f".{ext}" if dot else "")
                if stem and content_type:
                    found[stem] = StoredResource(key=key, content_type=content_type)
        return found

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def resource_key(resource_id: str, content_type: str) -> str:
        return f"{RESOURCE_PREFIX}{resource_id}{_EXT_BY_CT.get(content_type, '.bin')}"

    def _exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise


# Module-level singleton, same pattern as `manager` in the routes module.
storage = S3Storage()
