# Architecture

`thermal_tracker` is a Python package for tracking objects in thermal video. The current codebase keeps the existing algorithms intact and reorganizes them around runtime scenarios, processing stages, and external connections.

## Main Idea

The GUI is a dev workstation, not the center of the architecture. Frames, operator commands, processing scenarios, and results are separate parts so the same core can run in an interactive window today and in a headless Shared Memory runtime later.

## Runtime Layers

- `domain` contains shared data models and contracts.
- `scenarios` assemble processing stages into named modes such as `nn_manual` or `opencv_manual`.
- `processing_stages` contains algorithmic stages.
- `connections` contains frame readers, command readers, and result writers.
- `storage` contains history store interfaces and minimal placeholders.
- `gui` contains only the interactive dev UI.
- `config` loads algorithm presets and dev/runtime launch configs.

## Dev vs Runtime

Dev launch uses `presets/dev.toml`, `run_dev.py`, and the GUI. It is meant for research, visual inspection, manual target selection, and technical overlays.

Runtime launch uses `presets/runtime.toml`, `run_runtime.py`, and headless connections. Shared Memory belongs under:

- `thermal_tracker/connections/frames/shared_memory_frame_reader.py`
- `thermal_tracker/connections/commands/shared_memory_command_reader.py`
- `thermal_tracker/connections/results/shared_memory_result_writer.py`

The Shared Memory classes are placeholders for now. That is intentional: the architecture has a place for them, but the project does not pretend to have production IPC before it exists.

## Scenario Selection

`ScenarioFactory` creates scenarios by name from config:

- `nn_manual`
- `nn_auto`
- `opencv_manual`
- `opencv_auto_motion`

Legacy algorithm preset `kind` values are still mapped to scenarios for compatibility.
