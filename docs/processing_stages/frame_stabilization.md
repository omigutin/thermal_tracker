# Frame Stabilization

## За что отвечает стадия

Эта стадия пытается понять, насколько между кадрами сдвинулась вся картинка целиком.

Это особенно важно, потому что камера у нас может двигаться, и motion detector без этого часто начинает видеть не цель, а жизнь вообще.

## Что уже реализовано

Рабочие классы:
- `NoMotionEstimator`
- `PhaseCorrelationMotionEstimator`
- `FeatureAffineMotionEstimator`

`PhaseCorrelationMotionEstimator` сейчас используется в основном scenario.

## Что подготовлено на будущее

Заглушки:
- `EccTranslationMotionEstimator`
- `EccAffineMotionEstimator`
- `HomographyMotionEstimator`
- `TelemetryAssistedMotionEstimator`

## Когда что полезно

- `NoMotionEstimator` — baseline для сравнения.
- `PhaseCorrelation` — быстрый и практичный грубый сдвиг.
- `FeatureAffine` — когда нужен более взрослый вариант с точками и аффинной оценкой.
- `ECC` и `Homography` — более тяжёлые и точные варианты на будущее.
- `TelemetryAssisted` — для реальной платформы с внешними датчиками.

## Практический вывод

Стабилизация — не украшение, а реальный множитель качества для всех движущихся сцен.
