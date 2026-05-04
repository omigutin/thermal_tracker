# False Target Filtering

## За что отвечает стадия

После грубого детекта движения почти всегда появляются лишние кандидаты:
- шум;
- куски границ;
- объекты на краю кадра;
- слишком большие пятна;
- случайные вспышки.

Эта стадия чистит список кандидатов.

## Что уже реализовано

Рабочие классы:
- `FilterChain`
- `AreaAspectFalseTargetFilter`
- `BorderTouchFalseTargetFilter`
- `ContrastFalseTargetFilter`

## Что подготовлено на будущее

Заглушки:
- `PersistenceFalseTargetFilter`
- `MotionConsistencyFalseTargetFilter`
- `ClutterSuppressionFalseTargetFilter`

## Когда какой вариант полезен

- `AreaAspect` — выкинуть мусор по площади и вытянутости.
- `BorderTouch` — убрать всё, что прилипло к краям кадра.
- `Contrast` — проверить, отличается ли объект от локального фона.
- `Persistence` — фильтровать одноразовые вспышки.
- `MotionConsistency` — оставить только правдоподобные движения.
- `ClutterSuppression` — будущий режим для плотных сложных сцен.

## Практический вывод

Это очень полезная стадия для automatic scenario. Без неё auto-mode быстро начинает любить шум больше, чем реальные цели.
