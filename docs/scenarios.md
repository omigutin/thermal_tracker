# Сценарии

Сценарий — это готовая сборка стадий обработки и правил состояния.

Актуальные имена сценариев (по коду):

- `opencv_manual`
- `opencv_auto_motion`
- `nn_manual`
- `nn_auto`

Псевдонимы `pipeline.kind`, которые маппятся в эти сценарии:

- `manual_click_classical` -> `opencv_manual`
- `auto_motion_tracking` -> `opencv_auto_motion`
- `manual_click_neural` -> `nn_manual`
- `auto_neural_detection` -> `nn_auto`

Создание сценария выполняется через `ScenarioFactory`.
