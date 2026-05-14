from __future__ import annotations

import numpy as np

from thermal_tracker.core.config import load_app_config
from thermal_tracker.core.connections.commands.shared_memory_command_reader import SharedMemoryCommandReader
from thermal_tracker.core.connections.frames.shared_memory_frame_reader import SharedMemoryFrameReader
from thermal_tracker.core.connections.results.shared_memory_result_writer import SharedMemoryResultWriter
from thermal_tracker.core.connections.shared_memory import SharedMemoryFrameBuffer, SharedMemoryJsonBuffer
from thermal_tracker.core.storage.null_history_store import NullHistoryStore
from thermal_tracker.server.runtime_app import RuntimeApp
from thermal_tracker.server.services.gateway_service import GatewayConfig, SharedMemoryGateway
from thermal_tracker.server.services.shared_memory_cleanup import cleanup_shared_memory
from thermal_tracker.client.services.synthetic_network_sender import _build_frame


def test_shared_memory_buffers_roundtrip() -> None:
    prefix = "thermal_tracker_pytest_buffers"
    cleanup_shared_memory(prefix=prefix, camera_count=1)
    frame_buffer = SharedMemoryFrameBuffer(prefix=prefix, camera_id=0, width=512, height=640, channels=1, create=True)
    command_buffer = SharedMemoryJsonBuffer(prefix=prefix, kind="commands", create=True)
    try:
        frame = np.full((640, 512), 123, dtype=np.uint8)
        written = frame_buffer.write_frame(frame, frame_id=5, timestamp_ns=1000)
        assert written.frame_id == 5

        reader = SharedMemoryFrameReader(prefix=prefix, camera_id=0, width=512, height=640, channels=1)
        ok, read_frame = reader.read()
        assert ok
        assert read_frame is not None
        assert read_frame.shape == (640, 512)
        assert int(read_frame[10, 10]) == 123
        reader.close()

        command_buffer.write_message({"type": "reset"})
        command_reader = SharedMemoryCommandReader(prefix=prefix)
        assert command_reader.read() == {"type": "reset"}
        assert command_reader.read() is None
        command_reader.close()
    finally:
        frame_buffer.close()
        command_buffer.close()
        cleanup_shared_memory(prefix=prefix, camera_count=1)


def test_gateway_runtime_synthetic_click_roundtrip() -> None:
    prefix = "thermal_tracker_pytest_gateway"
    cleanup_shared_memory(prefix=prefix, camera_count=1)
    gateway = SharedMemoryGateway(GatewayConfig(prefix=prefix, width=512, height=640))
    app = None
    try:
        rng = np.random.default_rng(42)
        frame = _build_frame(
            frame_id=1,
            width=512,
            height=640,
            target_size=18,
            speed_x=1.8,
            speed_y=0.35,
            noise=0.0,
            rng=rng,
        )
        gateway.publish_raw_y8(frame.tobytes(), frame_id=1, remote_timestamp_ns=123)
        gateway.write_command({"type": "contrast_component", "x": 41, "y": 297})

        app = RuntimeApp(
            config=load_app_config("tests/fixtures/server_smoke.toml"),
            scenario_name="opencv_manual",
            scenario=None,
            frame_reader=SharedMemoryFrameReader(prefix=prefix, camera_id=0, width=512, height=640, channels=1),
            command_reader=SharedMemoryCommandReader(prefix=prefix),
            result_writer=SharedMemoryResultWriter(prefix=prefix),
            history_store=NullHistoryStore(),
        )
        result = app.process_once()
        assert result is not None
        assert result["frame_id"] == 1
        assert result["snapshot"]["state"] == "TRACKING"
        assert result["snapshot"]["bbox"] is not None

        metrics = gateway.metrics()
        assert metrics["frame"]["frame_id"] == 1
        assert metrics["result"]["frame_id"] == 1
        assert metrics["lag"]["frame_id_lag"] == 0
    finally:
        if app is not None:
            app.close()
        gateway.close()
        cleanup_shared_memory(prefix=prefix, camera_count=1)
