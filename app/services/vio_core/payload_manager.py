import os
import shutil
import numpy as np
from typing import Union

class VioPayloadManager:
    """Zero-Memory Payload Manager using np.memmap to stream intermediate VIO arrays to disk."""

    BASE_DIR = "/tmp/vio_sessions"

    @classmethod
    def get_session_dir(cls, session_id: str) -> str:
        """Return the disk directory path for a given session ID."""
        return os.path.join(cls.BASE_DIR, session_id)

    @classmethod
    def save_memmap_array(cls, session_id: str, key: str, array: np.ndarray) -> str:
        """Stream numpy array to disk as a memory-mappable .npy file under /tmp/vio_sessions/{session_id}/."""
        session_dir = cls.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)
        file_path = os.path.join(session_dir, f"{key}.npy")
        np.save(file_path, array)
        return file_path

    @classmethod
    def load_memmap_array(cls, session_id: str, key: str, mode: str = "r") -> np.memmap:
        """Load memory-mapped array from disk without allocating full python heap memory."""
        session_dir = cls.get_session_dir(session_id)
        file_path = os.path.join(session_dir, f"{key}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Memmap payload array key '{key}' not found for session '{session_id}'.")
        return np.load(file_path, mmap_mode=mode)

    @classmethod
    def cleanup_session(cls, session_id: str) -> None:
        """Remove session payload directory from disk."""
        session_dir = cls.get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
