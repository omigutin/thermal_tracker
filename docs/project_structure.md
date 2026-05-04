# Project Structure

```text
src/
  thermal_tracker/
    domain/
    scenarios/
    processing_stages/
    connections/
      frames/
      commands/
      results/
    nnet_interface/
    gui/
    storage/
    config/
    runtime_app.py
    cli.py

run_dev.py
run_runtime.py
models/
trackers/
presets/
configs/
video/
docs/
out/
```

## Processing Stages

- `frame_preprocessing` prepares frames.
- `frame_stabilization` estimates camera/global frame motion.
- `moving_area_detection` finds moving image areas.
- `target_candidate_extraction` extracts candidate objects.
- `candidate_filtering` filters false candidates.
- `target_selection` selects a target from operator input or candidates.
- `target_tracking` tracks the selected target.
- `target_recovery` is reserved for target recovery after target loss.

## Connections

- `connections/frames` reads frames from video, image sequences, RTSP, replay, or Shared Memory.
- `connections/commands` reads operator/runtime commands from GUI, Shared Memory, or null readers.
- `connections/results` writes results to GUI, Shared Memory, files, or logs.

Launch configs live in `configs/dev.toml` and `configs/runtime.toml`. Algorithm presets stay in `presets/`.
