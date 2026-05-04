# Tracking

## За что отвечает стадия

Tracking — это сердце ручного режима.

Задача стадии:
- взять уже выбранную цель;
- вести именно её по кадрам;
- переживать шум, изменение масштаба и краткие потери;
- не слишком охотно перескакивать на соседний объект.

## Что уже реализовано

Рабочие классы:
- `ClickToTrackSingleTargetTracker`
- `CsrtSingleTargetTracker`

Рабочие модели движения:
- `NoMotionModel`
- `ConstantVelocityMotionModel`
- `KalmanMotionModel`

## Что подготовлено на будущее

Заглушки:
- `TemplateSingleTargetTracker`
- `PointFlowSingleTargetTracker`
- `KcfSingleTargetTracker`
- `MosseSingleTargetTracker`
- `MedianFlowSingleTargetTracker`

## Как это читать правильно

- `ClickToTrackSingleTargetTracker` — главный текущий гибридный трекер.
  Он сочетает внешний вид цели, локальные точки и внутреннюю reacquire-логику.
- `CsrtSingleTargetTracker` — сильный baseline из OpenCV для сравнения.
- Модели движения лежат отдельно, потому что `Kalman` — это не самостоятельный tracker, а помощник для прогноза.

## Практический вывод

Сейчас именно tracking плюс click initialization дают основной результат проекта. Всё, что улучшает эти две стадии, почти всегда сразу видно на видео.
