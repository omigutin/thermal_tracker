# Motion Detection

## За что отвечает стадия

Эта стадия пытается найти в кадре всё, что похоже на движение.

В ручном режиме она не обязательна, но в автоматическом режиме без неё никуда.

## Что уже реализовано

Рабочие классы:
- `FrameDifferenceMotionDetector`
- `RunningAverageMotionDetector`
- `Mog2MotionDetector`
- `KnnMotionDetector`

## Что подготовлено на будущее

Заглушки:
- `OpticalFlowMotionDetector`
- `ThermalChangeMotionDetector`

## Когда какой вариант полезен

- `FrameDifference` — самый простой baseline.
- `RunningAverage` — когда фон меняется медленно.
- `MOG2` — сильный практический baseline OpenCV.
- `KNN` — ещё один полезный baseline для сравнения.
- `OpticalFlow` — когда движение лучше видно по полю смещений.
- `ThermalChange` — когда надо мыслить уже более thermal-aware логикой.

## Практический вывод

Motion detection важен для автоматического режима, но для текущего click-to-track он пока не главный герой.
