# Frame Preprocessing

## За что отвечает стадия

Предобработка делает кадр более удобным для следующих этапов.

Что сюда обычно входит:
- перевод в серый формат;
- шумоподавление;
- нормализация яркости;
- усиление локального контраста;
- построение карты градиентов.

## Что уже реализовано

Рабочие классы:
- `IdentityFramePreprocessor`
- `PercentileNormalizePreprocessor`
- `ClaheContrastPreprocessor`
- `BilateralFramePreprocessor`
- `ThermalFramePreprocessor`

`ThermalFramePreprocessor` сейчас главный рабочий вариант. Это составной практичный препроцессор под текущий контур.

## Что подготовлено как задел

Заглушки:
- `TemporalDenoisePreprocessor`
- `GradientEnhancedPreprocessor`
- `AgcCompensationPreprocessor`

## Когда какой вариант полезен

- `Identity` — baseline и отладка.
- `PercentileNormalize` — когда min-max портится выбросами.
- `ClaheContrast` — когда надо вытянуть локальный контраст.
- `Bilateral` — когда надо шум приглушить, но границы пожалеть.
- `ThermalFramePreprocessor` — основной сбалансированный вариант для текущего трекера.

## Практический вывод

Предобработка очень сильно влияет на качество клика, маски движения и устойчивость трекера. Это одна из самых чувствительных стадий проекта.
