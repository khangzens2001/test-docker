import hashlib
import os
import struct


def verify_weight_integrity(model_bytes: bytes, expected_sha: str | None = None) -> bool:
    """Verify SHA-256 digest of model_bytes matches expected_sha.

    Returns True if expected_sha is None or empty, or if computed hash matches expected_sha.
    Returns False otherwise.
    """
    if not expected_sha:
        return True
    computed_sha = hashlib.sha256(model_bytes).hexdigest()
    return computed_sha == expected_sha


# --- Image header verification ---

# Cap how many bytes we scan while walking JPEG markers to bound worst case work.
_JPEG_MAX_SCAN_BYTES = 256 * 1024
_JPEG_MAX_MARKERS = 512
_MAX_IMAGE_DIM = 4096


def verify_image_headers(file_path: str) -> tuple[bool, str | None]:
    """Validate that `file_path` is a PNG or JPEG within dimension limits.

    Returns (True, None) on success, (False, reason) otherwise. Never raises.
    """
    try:
        size = os.path.getsize(file_path)
        if size < 24:
            return False, "File too small to be an image."

        with open(file_path, "rb") as f:
            header = f.read(8)

            # PNG: 89 50 4E 47 0D 0A 1A 0A + IHDR
            if header == b"\x89PNG\r\n\x1a\n":
                ihdr = f.read(21)
                if len(ihdr) < 16:
                    return False, "Malformed PNG header."
                width = struct.unpack(">I", ihdr[8:12])[0]
                height = struct.unpack(">I", ihdr[12:16])[0]
                if width > _MAX_IMAGE_DIM or height > _MAX_IMAGE_DIM:
                    return (
                        False,
                        f"PNG size {width}x{height} exceeds limit of {_MAX_IMAGE_DIM}.",
                    )
                return True, None

            # JPEG: starts with FF D8
            if header[:2] == b"\xff\xd8":
                f.seek(2)
                scanned = 2
                markers_seen = 0
                while (
                    scanned < min(size, _JPEG_MAX_SCAN_BYTES)
                    and markers_seen < _JPEG_MAX_MARKERS
                ):
                    marker_meta = f.read(4)
                    scanned += 4
                    if len(marker_meta) < 4:
                        return False, "Malformed JPEG structure."
                    marker, segment_len = struct.unpack(">HH", marker_meta)
                    markers_seen += 1

                    # SOF0/SOF1/SOF2/SOF3/etc. carry frame dimensions.
                    if marker in (
                        0xFFC0,
                        0xFFC1,
                        0xFFC2,
                        0xFFC3,
                        0xFFC5,
                        0xFFC6,
                        0xFFC7,
                        0xFFC9,
                        0xFFCA,
                        0xFFCB,
                        0xFFCD,
                        0xFFCE,
                        0xFFCF,
                    ):
                        sof = f.read(5)
                        scanned += 5
                        if len(sof) < 5:
                            return False, "Malformed SOF segment."
                        # Layout: 1 byte precision, 2 bytes height, 2 bytes width
                        height, width = struct.unpack(">HH", sof[1:5])
                        if width > _MAX_IMAGE_DIM or height > _MAX_IMAGE_DIM:
                            return (
                                False,
                                f"JPEG size {width}x{height} exceeds limit "
                                f"of {_MAX_IMAGE_DIM}.",
                            )
                        return True, None

                    if segment_len < 2:
                        return False, "Invalid JPEG segment length."
                    f.seek(segment_len - 2, os.SEEK_CUR)
                    scanned += segment_len - 2
                return False, "JPEG SOF marker not found within scan window."

            return False, "Invalid file format: must be JPEG or PNG."
    except Exception as exc:
        return False, f"Image processing error: {exc}"

