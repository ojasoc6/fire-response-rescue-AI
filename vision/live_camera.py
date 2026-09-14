"""Live camera abstraction for real‑time video streams.

Provides a :class:`LiveCamera` that abstracts OpenCV ``VideoCapture`` handling,
supports optional DirectShow fallback, resolution negotiation, FPS tracking,
and context‑manager usage. The class is deliberately lightweight – it does **not**
perform any per‑frame disk I/O.
"""

import cv2
import time
from typing import Tuple

class LiveCamera:
    """Encapsulate a video capture source with FPS and resolution reporting.

    Parameters
    ----------
    source: int | str, optional
        Index of the camera (default ``1`` – typical for DroidCam) or a URL.
    width: int, optional
        Desired frame width; the actual width may differ based on hardware.
    height: int, optional
        Desired frame height.
    """

    def __init__(self, source: int | str = 1, width: int = 1920, height: int = 1080):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self._fps = 0.0
        self._last_ts = None
        self._resolution = (0, 0)
        self._open()

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _open(self) -> None:
        """Attempt to open the capture device.
        Falls back to ``CAP_DSHOW`` on Windows if the default backend fails.
        """
        # First try default backend.
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            # Fallback for Windows – DirectShow may be required for virtual cams.
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source {self.source!r}")

        # Request resolution – the driver may clamp to supported sizes.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (actual_w, actual_h)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @property
    def resolution(self) -> Tuple[int, int]:
        """Current frame resolution ``(width, height)``.
        """
        return self._resolution

    @property
    def fps(self) -> float:
        """Smoothed frames‑per‑second estimate based on the last two frames.
        """
        return self._fps

    def read(self) -> Tuple[bool, "any"]:
        """Read the next frame.

        Returns ``(ret, frame)`` where ``ret`` is ``True`` on success. The method
        also updates the FPS estimate.
        """
        ret, frame = self.cap.read()
        if ret:
            now = time.time()
            if self._last_ts is not None:
                delta = now - self._last_ts
                self._fps = 1.0 / delta if delta > 0 else 0.0
            self._last_ts = now
        return ret, frame

    def release(self) -> None:
        """Release the underlying ``VideoCapture`` object.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------------------------------------------------------------------
    # Context‑manager support – enables ``with LiveCamera() as cam:`` syntax.
    # ---------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # Do not suppress exceptions.
        return False
