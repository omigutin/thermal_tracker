"""Источники кадров.

Здесь собраны как рабочие реализации, так и заготовки под будущие режимы.
"""

from .base_frame_reader import BaseFrameSource
from .image_sequence_frame_reader import ImageSequenceFrameSource
from .multi_camera_frame_reader import MultiCameraFrameSource
from .opencv_video_frame_reader import OpenCvVideoSource
from .replay_frame_reader import ReplayFrameSource
from .rtsp_stream_frame_reader import RtspStreamFrameSource
from .shared_memory_frame_reader import SharedMemoryFrameReader

BaseFrameReader = BaseFrameSource
ImageSequenceFrameReader = ImageSequenceFrameSource
MultiCameraFrameReader = MultiCameraFrameSource
OpenCvVideoFrameReader = OpenCvVideoSource
ReplayFrameReader = ReplayFrameSource
RtspStreamFrameReader = RtspStreamFrameSource


def create_frame_reader(config):
    reader = (getattr(config, "reader", "") or "").strip()
    source_path = getattr(config, "source_path", "")
    if reader == "opencv_video":
        return OpenCvVideoSource(source_path)
    if reader == "image_sequence":
        return ImageSequenceFrameSource(source_path)
    if reader == "rtsp_stream":
        return RtspStreamFrameSource(source_path)
    if reader == "shared_memory":
        return SharedMemoryFrameReader(getattr(config, "camera_count", 1))
    if reader == "multi_camera":
        return MultiCameraFrameSource([])
    raise ValueError(f"Unknown frame reader: {reader!r}")

__all__ = [
    "BaseFrameReader",
    "BaseFrameSource",
    "ImageSequenceFrameReader",
    "ImageSequenceFrameSource",
    "MultiCameraFrameReader",
    "MultiCameraFrameSource",
    "OpenCvVideoFrameReader",
    "OpenCvVideoSource",
    "ReplayFrameReader",
    "ReplayFrameSource",
    "RtspStreamFrameReader",
    "RtspStreamFrameSource",
    "SharedMemoryFrameReader",
    "create_frame_reader",
]
