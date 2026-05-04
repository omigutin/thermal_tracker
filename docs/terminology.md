# Terminology

`candidate` means "кандидат на цель" before operator choice or confirmation.

`target` means "выбранная / сопровождаемая цель" after confirmation.

`NN` means Neural Network, "нейронная сеть".

`OpenCV` means Open Source Computer Vision Library.

## Naming Rules

- Use `candidate` before target selection.
- Use `target` after a target is selected for tracking.
- Use `nn_` for neural-network implementations and scenarios.
- Use `nnet_interface/` for the temporary NN integration interface.
- Use `opencv_` only for implementations that directly use OpenCV or wrap an OpenCV algorithm.
- Use `base_` for abstract contracts.
- Use `no_` for no-op implementations.
- Use `identity_` for pass-through implementations.
