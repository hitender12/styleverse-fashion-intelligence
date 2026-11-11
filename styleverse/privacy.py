"""Privacy controls for uploads.

- Uploads are session-scoped: stored under a session directory with a
  time-to-live; expired sessions can be deleted in full.
- EXIF metadata (location, device info) is stripped on ingest: the stored
  image is re-encoded without any EXIF segment.
- File-size and MIME-type checks reject oversized or non-image uploads.
"""

from __future__ import annotations

import io
import shutil
import time
from pathlib import Path

from PIL import Image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
SESSION_TTL_SECONDS = 24 * 3600  # 24 hours


class UploadError(ValueError):
    """Raised when an upload fails validation."""


def strip_exif(image_bytes: bytes) -> bytes:
    """Re-encode an image with no EXIF segment. Returns JPEG bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.format not in ALLOWED_FORMATS and img.format is not None:
        raise UploadError(f"unsupported image format: {img.format}")
    out = io.BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue()


def ingest_upload(image_bytes: bytes, session_dir: str | Path,
                  filename: str = "query.jpg") -> Path:
    """Validate, strip EXIF, and store an upload inside a session dir."""
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"upload too large: {len(image_bytes)} bytes "
            f"(max {MAX_UPLOAD_BYTES})")
    if len(image_bytes) == 0:
        raise UploadError("empty upload")
    clean = strip_exif(image_bytes)
    path = Path(session_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    target.write_bytes(clean)
    return target


class Session:
    """A search session: uploads, filters, feedback, and an expiry."""

    def __init__(self, session_id: str, base_dir: str | Path = "sessions",
                 ttl_seconds: int = SESSION_TTL_SECONDS):
        self.session_id = session_id
        self.dir = Path(base_dir) / session_id
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.dir.mkdir(parents=True, exist_ok=True)

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) - self.created_at \
            > self.ttl_seconds

    def delete(self) -> None:
        """Delete the session directory and everything in it."""
        shutil.rmtree(self.dir, ignore_errors=True)

    def upload(self, image_bytes: bytes, filename: str = "query.jpg") -> Path:
        if self.expired():
            raise UploadError("session expired")
        return ingest_upload(image_bytes, self.dir, filename)
