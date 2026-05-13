# Контекст: баги IRST-трекера в web-клиенте

Документ создан для анализа двух живых багов IRST-трекера в более умной модели.
Содержит: полный путь данных, все ключевые файлы с кодом, телеметрию и гипотезы.

---

## 1. Контекст проекта

**Проект**: thermal_tracker — Python-бэкенд трекинга тепловых целей.

**IRST** — Infrared Search and Track / инфракрасный поиск и сопровождение.
Детектор локального контраста + фильтр Калмана, без template matching и LK optical flow.

**Пресет**: `irst_small_target` — для маленьких тепловых целей.

---

## 2. Архитектура системы (полный путь кадра и клика)

```
Браузер (app.js)
  │
  ├── Загрузка кадра:
  │     sourceVideo (1280×1024) → containRect → uploadCanvas (512×640)
  │     [lastUploadContentRect = {x:0, y:115, w:512, h:410}]
  │     POST /api/frames/raw-y8?content_x=0&content_y=115&... (body: 512×640 raw Y8 bytes)
  │
  ├── Отображение:
  │     GET /api/frame/latest.jpg → displayCanvas (512×410)
  │     [crop rows 115..525 из 512×640 JPEG]
  │
  └── Клик:
        onCanvasClick → {x, y} в frame-координатах
        POST /api/commands/click {x, y}

Gateway (gateway_service.py)
  │
  ├── Входящий кадр → SharedMemoryFrameBuffer (512×640 raw Y8)
  ├── Команда click → SharedMemoryJsonBuffer {type:"click", x, y}
  └── GET latest.jpg → берёт кадр из SharedMemory, рисует overlay из result

Runtime (runtime_app.py)
  │
  ├── Читает кадр из SharedMemory (512×640)
  ├── Читает команду из SharedMemory
  └── _apply_runtime_command: pending_click = (x, y)

Scenario (opencv_manual_scenario.py → ManualClickTrackingPipeline)
  │
  ├── process_next_raw_frame(raw_512x640, runtime)
  │     self.current_frame = self.preprocessor.process(raw_512x640)
  │     [Preprocessor: resize → gaussian_blur → median_blur → normalize → clahe → sharpness]
  │     [resize_width=960, но 512 ≤ 960 → resize — NO-OP → current_frame остаётся 512×640]
  │
  ├── На клик: _on_click(self.current_frame, pending_click)
  │     tracker.start_tracking(frame_512x640, point=(x, y))
  │
  └── На обычный кадр: tracker.update(frame_512x640, motion)

Tracker (irst_contrast_target_tracker.py → IrstSingleTargetTracker)
  │
  ├── start_tracking(frame, point) → ищет blob в click_search_radius=50px вокруг point
  └── update(frame, motion) → Kalman predict + _find_candidates + _associate
```

---

## 3. Критическое открытие: resize — NO-OP

**Файл**: `src/thermal_tracker/core/processing_stages/frame_preprocessing/resize_frame_preprocessor.py`

```python
def process(self, frame: ProcessedFrame) -> ProcessedFrame:
    target_width = self._target_width
    if target_width is None or frame.bgr.shape[1] <= target_width:
        return frame  # ← НЕ МАСШТАБИРУЕТ, если уже ≤ target_width
    # ...
```

Пресет: `resize_width = 960`. Входящий кадр: 512 пикселей шириной.
**512 ≤ 960 → resize ничего не делает. Processed frame остаётся 512×640.**

Вывод: пространство координат единое во всём pipeline: 512×640.

---

## 4. Анализ координат клика в браузере

**Функция** `onCanvasClick` в `src/thermal_tracker/client/web/app.js`:

```javascript
async function onCanvasClick(event) {
    const rect = displayCanvas.getBoundingClientRect();  // CSS-размер канваса
    const scaleX = displayCanvas.width / rect.width;    // 512 / css_width
    const scaleY = displayCanvas.height / rect.height;  // 410 / css_height
    const x = Math.round(displayedContentRect.x + (event.clientX - rect.left) * scaleX);
    const y = Math.round(displayedContentRect.y + (event.clientY - rect.top) * scaleY);
    // При корректном displayedContentRect = {x:0, y:115, w:512, h:410}:
    // x = 0 + local_x * (512/css_width)       → frame x ∈ [0..512]
    // y = 115 + local_y * (410/css_height)     → frame y ∈ [115..525]
    await sendCurrentFrame();
    await fetch("/api/commands/click", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y }),
    });
    await sendCurrentFrame();
    window.setTimeout(refreshFrame, 120);
    window.setTimeout(refreshMetrics, 180);
}
```

**При корректном `displayedContentRect`**: формула правильная, Y-смещение 115px учтено.

---

## 5. Откуда берётся displayedContentRect и displayCanvas.height

**Функция** `refreshFrame()` в `app.js`:

```javascript
async function refreshFrame() {
    const image = new Image();
    image.onload = () => {
        displayedContentRect = clampRect(lastUploadContentRect, image.naturalWidth, image.naturalHeight);
        // clampRect({x:0, y:115, w:512, h:410}, 512, 640) = {x:0, y:115, w:512, h:410}
        if (displayCanvas.width !== displayedContentRect.width ||
            displayCanvas.height !== displayedContentRect.height) {
            displayCanvas.width = displayedContentRect.width;   // 512
            displayCanvas.height = displayedContentRect.height; // 410
        }
        displayContext.drawImage(image,
            0, 115, 512, 410,   // src: вырезаем строки 115..525 из 512×640 JPEG
            0, 0, 512, 410      // dst: весь 512×410 displayCanvas
        );
        fitDisplayCanvasToStage();
    };
    image.src = `/api/frame/latest.jpg?overlay=true&t=${Date.now()}`;
}
```

**Начальные значения** (до первого image.onload):
```javascript
let lastUploadContentRect = { x: 0, y: 0, width: TARGET_WIDTH, height: TARGET_HEIGHT };
// = {x:0, y:0, width:512, height:640}
let displayedContentRect = lastUploadContentRect;
// = {x:0, y:0, width:512, height:640}  ← НАЧАЛЬНОЕ: Y-смещение = 0, не 115!
```

**`lastUploadContentRect` обновляется** в `drawSourceVideoToUploadCanvas()`:
```javascript
function drawSourceVideoToUploadCanvas() {
    // containRect(1280, 1024, 512, 640):
    //   scale = min(512/1280, 640/1024) = 0.4
    //   width=512, height=410, x=0, y=115
    const rect = containRect(sourceVideo.videoWidth, sourceVideo.videoHeight, TARGET_WIDTH, TARGET_HEIGHT);
    lastUploadContentRect = rect;  // = {x:0, y:115, w:512, h:410}
    // ...
}
```

`drawSourceVideoToUploadCanvas()` вызывается внутри `sendCurrentFrame()`.

**`displayedContentRect` обновляется** только когда image.onload срабатывает в `refreshFrame()`.

---

## 6. Гипотеза 1: Race condition в displayedContentRect

**Сценарий** (возможная причина "обводка не там где кликаю"):

```
loadSelectedVideo()
  → sendCurrentFrame()
       → drawSourceVideoToUploadCanvas()  [lastUploadContentRect = {x:0,y:115,...}]
       → POST /api/frames/raw-y8
       → refreshFrame()                  [запускает async image.src = .../latest.jpg]
  ← возвращает управление в браузер
                                         [image всё ещё грузится]
  ← ПОЛЬЗОВАТЕЛЬ КЛИКАЕТ
       onCanvasClick: displayedContentRect = {x:0, y:0, w:512, h:640}  ← НАЧАЛЬНОЕ!
       y = 0 + local_y * (640/css_h)     ← нет смещения 115px!
       Пример: click display_y=200 → frame_y=200 (НЕВЕРНО, должно быть 315)
  ← image.onload срабатывает
       displayedContentRect = {x:0, y:115, w:512, h:410}  ← теперь правильно
       displayCanvas.height = 410
```

**Эффект при стейле `displayedContentRect.y=0`**:
- `scaleY = 640/css_height` вместо `410/css_height`
- frame_y = 0 + local_y * (640/css_h) ← без смещения 115, другой масштаб
- Трекер ищет blob в (x, y_wrong) → blob не найден → fallback bbox в (x, y_wrong)
- Fallback bbox отображается не там, где кликнул пользователь

**Частота**: баг проявляется при быстром клике сразу после выбора файла,
но НЕ проявляется при кликах во время воспроизведения (displayedContentRect уже обновлён).

**Проверка**: добавить `console.log(displayedContentRect)` в `onCanvasClick`.

---

## 7. Гипотеза 2: Overlay рисуется в правильных координатах?

**Overlay** рисует `gateway_service.py → _draw_result_overlay`:
```python
def _draw_result_overlay(image: np.ndarray, result: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 512×640
    # ...
    _draw_bbox(canvas, snapshot.get("bbox"), (0, 255, 0))
    # bbox.x, bbox.y — в координатах processed_frame (512×640, т.к. resize=NO-OP)
```

`/api/frame/latest.jpg` возвращает кадр из `SharedMemory` — это **raw_y8 frame 512×640**.
Overlay bbox рисуется в 512×640 координатах на 512×640 изображении. **Это корректно**.

Браузер кропит строки [115..525] → bbox в y=340 отображается в display_y=340-115=225. **Корректно**.

**Вывод**: сама overlay-отрисовка правильная, **но только если bbox.y приходит в 512×640 пространстве**.
Поскольку resize=NO-OP, processed frame = 512×640, bbox.y будет в 512×640 → overlay корректен.

---

## 8. Анализ start_tracking (что происходит при клике)

**Файл**: `src/thermal_tracker/core/processing_stages/target_tracking/irst_contrast_target_tracker.py`

```python
def start_tracking(self, frame: ProcessedFrame, point: tuple[int, int]) -> TrackSnapshot:
    search_region = BoundingBox.from_center(
        point[0], point[1],
        self.config.click_search_radius * 2,  # 100px × 100px зона
        self.config.click_search_radius * 2,
    ).clamp(frame.bgr.shape)

    candidates = self._find_candidates(frame, search_region)

    if candidates:
        # ИСПРАВЛЕНИЕ (сделано в предыдущей сессии):
        # Раньше: min(distance) — выбирал ближайший blob, мог выбрать шум
        # Теперь: max(score / (1 + dist/ref)) — взвешенный критерий
        reference_dist = max(self.config.click_search_radius / 4.0, 1.0)  # = 12.5px
        best_bbox, best_score = max(
            candidates,
            key=lambda c: c[1] / (
                1.0 + math.hypot(c[0].center[0] - point[0], c[0].center[1] - point[1]) / reference_dist
            ),
        )
    else:
        # Fallback: bbox фиксированного размера ВОКРУГ ТОЧКИ КЛИКА
        s = self.config.click_fallback_size  # = 10px
        best_bbox = BoundingBox.from_center(point[0], point[1], s, s).clamp(frame.bgr.shape)
        best_score = 0.0

    # ... инициализация трекера ...
    self._state = TrackerState.TRACKING  # всегда TRACKING, даже для fallback
    return self.snapshot(GlobalMotion())
```

**Важно**: даже при fallback (blob не найден) трекер уходит в `TRACKING`.
Fallback bbox центрирован точно на `point` — то есть там, где кликнул пользователь.

Если `point` неверный (из-за race condition), fallback bbox будет в неверном месте.

---

## 9. Анализ качества детекции (низкие scores)

**Из телеметрии второго теста**:
- Все 14 треков имеют score ∈ [0.041..0.058]
- `contrast_threshold = 12.0`
- Score вычисляется как: `mean(contrast_map at blob pixels) / 255`
- Score = 0.047 → mean(contrast) ≈ 12.0 — ровно на пороге!

Цель балансирует на пороге детекции. Любой шум в кадре может:
- Уронить контраст ниже порога → blob не детектируется → трек уходит в SEARCHING
- Создать ложный blob рядом → трек захватывает ложный blob

**Возможное улучшение**: снизить `contrast_threshold` с 12.0 до 8.0–10.0.

---

## 10. Анализ физического gate в _associate

**Файл**: `irst_contrast_target_tracker.py`, метод `_associate`:

```python
def _associate(self, candidates, predicted_center, last_bbox, lost_frames):
    gate_radius = min(
        self.config.min_gate + lost_frames * self.config.gate_growth,
        self.config.max_gate,
    )
    # lost_frames=0: gate_radius = 20px

    if last_bbox is not None:
        max_physical_dist = self.config.max_speed_px_per_frame * max(lost_frames, 1) * 1.5
        # lost_frames=0: max(0, 1)=1 → max_physical_dist = 6.0 * 1 * 1.5 = 9.0px!
        last_cx, last_cy = last_bbox.center

    for bbox, score in candidates:
        cx, cy = bbox.center
        dist_from_pred = math.hypot(cx - px, cy - py)
        dist_from_last = math.hypot(cx - last_cx, cy - last_cy)

        if dist_from_pred > gate_radius:    # > 20px → reject
            continue
        if dist_from_last > max_physical_dist:  # > 9.0px → reject на первом update!
            continue

        if dist_from_pred < best_dist:  # выбираем ближайший к Kalman-прогнозу
            best = (bbox, score)
```

**Потенциальная проблема**: На ПЕРВОМ `update()` после `start_tracking()`:
- `lost_frames = 0`
- `max_physical_dist = 6.0 * max(0, 1) * 1.5 = 9.0px`
- Любое смещение blob > 9px от его позиции на первом кадре → трек теряется!

При движении цели 6px/кадр это OK (укладывается в 9px). Но если:
- Blob шумит и его центроид прыгает (нестабильный детектор)
- Цель движется быстрее 6px/кадр
- Первый клик попал в fallback (blob не найден), и `canonical_size` неточный

Трек может потерять цель на втором кадре.

**Критично**: `_associate` выбирает кандидата по `min(dist_from_pred)`, а не по взвешенному
score/distance, как `start_tracking`. Это разные стратегии выбора!

---

## 11. Параметры пресета (irst_small_target.toml)

```toml
[irst_tracking]
filter_kernel = 3        # внутреннее окно детектора (объект)
background_kernel = 11   # окно фона
contrast_threshold = 12.0  # порог, на котором балансирует цель (score ≈ 0.047!)
min_blob_area = 1
max_blob_area = 200

kalman_process_noise_pos = 0.5
kalman_process_noise_vel = 2.0
kalman_measurement_noise = 1.5

min_gate = 20        # начальный gate
gate_growth = 3      # рост gate за кадр потери
max_gate = 100
max_speed_px_per_frame = 6.0  # физический gate

max_lost_frames = 30
blur_hold_enabled = true
blur_sharpness_drop_ratio = 0.60

click_search_radius = 50   # радиус поиска blob вокруг клика
click_fallback_size = 10   # размер fallback-bbox

[preprocessing]
methods = ["resize", "gaussian_blur", "median_blur", "minmax_normalize",
           "clahe_contrast", "sharpness_metric"]
resize_width = 960   # ← NO-OP: 512 ≤ 960, кадр не масштабируется!
gaussian_kernel = 3
median_kernel = 3
clahe_clip_limit = 2.0
clahe_tile_grid_size = 8
```

---

## 12. Параметры кадра в тесте (из JSONL-телеметрии)

```
source: 1280×1024 → браузер → 512×640 raw Y8
frame: width=512, height=640, format=raw_y8
content_rect: {x:0, y:115, width:512, height:410}
  (видео размещено в y=[115..525], чёрные полосы сверху и снизу)

14 треков (ID 0..13), большинство живут 2 кадра
Все первые bbox.y ∈ [289..381] (frame-координаты)
Все score ∈ [0.041..0.058] — на пороге detektsii

Трек #3 (самый долгий, 33 кадра):
  TRACKING → SEARCHING → RECOVERING
  В RECOVERING bbox скачет между y=318 и y=343 (25px разброс)
  Затем уходит в x=165,y=333 и x=189,y=330 (большой скачок)
```

---

## 13. Что было изменено (до второго теста)

### Fix 1: start_tracking — критерий выбора blob
**Файл**: `irst_contrast_target_tracker.py`

```python
# БЫЛО (простой min-distance):
best_bbox, best_score = min(
    candidates,
    key=lambda c: math.hypot(c[0].center[0] - point[0], c[0].center[1] - point[1]),
)

# СТАЛО (взвешенный score/distance):
reference_dist = max(self.config.click_search_radius / 4.0, 1.0)  # 12.5px
best_bbox, best_score = max(
    candidates,
    key=lambda c: c[1] / (
        1.0 + math.hypot(c[0].center[0] - point[0], c[0].center[1] - point[1]) / reference_dist
    ),
)
```

### Fix 2: физический gate в IrstContrastRecoverer
**Файл**: `irst_contrast_target_recoverer.py`

Добавлен параметр `max_speed_px_per_frame = 6.0` в `__init__`.
В `reacquire()` добавлен физический gate и взвешенная выборка (аналогично start_tracking).
Параметр передаётся из `TargetRecoveryConfig.max_speed_px_per_frame`.

### Fix 3: TargetRecoveryConfig — новое поле
**Файл**: `preset.py`

```python
@dataclass
class TargetRecoveryConfig:
    # ...
    max_speed_px_per_frame: float = 0.0  # 0.0 = gate выключен (совместимость)
```

В `irst_small_target.toml`: `max_speed_px_per_frame = 6.0` в секции `[target_recovery]`.

---

## 14. Что нужно проверить аналитику (приоритеты)

### ВЫСШИЙ ПРИОРИТЕТ

**Гипотеза A: Race condition в displayedContentRect**

Вопрос: в момент клика (`onCanvasClick`) успевает ли `refreshFrame()` получить ответ
от сервера и обновить `displayedContentRect`?

- Если пользователь кликает сразу после выбора файла, `displayedContentRect.y = 0` (начальное),
  а не 115. Это сдвигает frame_y клика на −115px и меняет scaleY.
- Добавить `console.log("click:", x, y, "displayedContentRect:", displayedContentRect)`
  в начало `onCanvasClick` и проверить в DevTools консоли.

**Гипотеза B: displayedContentRect.y правильный, но target не детектируется**

Если `displayedContentRect` корректен, click_coords попадают в нужную область,
но blob не найден → fallback bbox создаётся точно на click point → отображается там, где кликнул.
В этом случае проблема не в координатах, а в низком контрасте (score ≈ threshold).

Проверить: снизить `contrast_threshold` с 12.0 до 8.0. Если треки стабилизируются → причина найдена.

### СРЕДНИЙ ПРИОРИТЕТ

**Гипотеза C: _associate слишком строгий при lost_frames=0**

На первом `update()` после `start_tracking`, `max_physical_dist = 9.0px`.
Если blob-центроид нестабильный и прыгает на >9px между кадрами,
трек теряется на кадре 2 из 2.

Проверить: временно увеличить `max_speed_px_per_frame` с 6.0 до 12.0
или добавить `min_physical_dist_for_lost_0` как отдельный параметр.

**Гипотеза D: CLAHE создаёт артефакты на чёрных полосах**

CLAHE с `tile_grid_size=8` применяется ко всему 512×640 кадру, включая чёрные полосы y=[0..115] и y=[525..640].
Артефакты в чёрных полосах могут создавать ложные blob-ы.
Но `click_search_radius=50` ограничивает поиск — только если клик сам оказался в неверной y-зоне
(из-за race condition), blob из чёрной полосы мог быть выбран.

---

## 15. Полная структура файлов

```
src/thermal_tracker/
├── client/
│   └── web/
│       └── app.js                  ← click mapping, display, onCanvasClick
├── server/
│   └── services/
│       └── gateway_service.py      ← /api/commands/click, overlay drawing, latest.jpg
├── server/
│   └── runtime_app.py              ← _apply_runtime_command → pending_click
├── core/
│   ├── config/
│   │   └── preset.py               ← TargetRecoveryConfig (max_speed_px_per_frame добавлен)
│   ├── scenarios/
│   │   └── opencv_manual_scenario.py ← ManualClickTrackingPipeline._on_click
│   └── processing_stages/
│       ├── frame_preprocessing/
│       │   └── resize_frame_preprocessor.py  ← NO-OP при w≤960!
│       ├── target_tracking/
│       │   └── irst_contrast_target_tracker.py ← start_tracking, update, _associate
│       └── target_recovery/
│           └── irst_contrast_target_recoverer.py ← reacquire (физический gate добавлен)
└── presets/
    └── irst_small_target.toml      ← contrast_threshold=12.0, click_search_radius=50
```

---

## 16. Рекомендуемые действия для диагностики

1. **Добавить console.log в onCanvasClick** (app.js):
   ```javascript
   console.log("click at", x, y, "displayedContentRect:", JSON.stringify(displayedContentRect),
               "canvas:", displayCanvas.width, displayCanvas.height,
               "css:", displayCanvas.getBoundingClientRect().width, displayCanvas.getBoundingClientRect().height);
   ```
   Открыть DevTools → Console → кликнуть на объект → посмотреть реальные координаты.

2. **Если displayedContentRect.y=0 (race condition)**:
   Фикс: в `onCanvasClick` использовать `lastUploadContentRect` вместо `displayedContentRect`,
   или инициализировать `displayedContentRect = { x: 0, y: 0, width: TARGET_WIDTH, height: TARGET_HEIGHT }` ← уже так,
   НО инициализировать его из данных видео СИНХРОННО при `loadSelectedVideo`,
   сразу после `drawSourceVideoToUploadCanvas()` устанавливать и `displayedContentRect = lastUploadContentRect`.

3. **Если displayedContentRect.y=115 (race condition НЕ причина)**:
   - Проверить `contrast_threshold` → снизить до 8.0
   - Добавить логирование в трекер: сколько кандидатов найдено при клике,
     на каком расстоянии от point, какой score выбранного blob
   - Если 0 кандидатов: причина — порог слишком высокий или CLAHE недостаточный
   - Если >0 кандидатов, но выбирается не тот: причина — в логике взвешивания

4. **Tracing через JSONL**: найти строки с `event: "frame"` и `snapshot.state: "TRACKING"`,
   посмотреть `snapshot.bbox` относительно `content_rect` и `click_requested` события.
   Если JSONL содержит event=click_requested (для desktop-клиента) — сравнить click coordinates с bbox.

---

## 17. Точка входа для быстрой проверки (без запуска сервера)

```python
# test_click_coords.py — запустить в любом Python 3.12+
# Воспроизводит полный coordinate mapping для точки клика

TARGET_WIDTH = 512
TARGET_HEIGHT = 640

def contain_rect(src_w, src_h, tgt_w, tgt_h):
    scale = min(tgt_w / src_w, tgt_h / src_h)
    w = max(1, round(src_w * scale))
    h = max(1, round(src_h * scale))
    return {
        "x": (tgt_w - w) // 2,
        "y": (tgt_h - h) // 2,
        "width": w, "height": h,
    }

# Источник 1280×1024
cr = contain_rect(1280, 1024, 512, 640)
print("content_rect:", cr)   # {x:0, y:115, w:512, h:410}

# Пользователь кликает по центру контентной области
display_click_x = cr["width"] // 2     # 256 (CSS пикселей)
display_click_y = cr["height"] // 2    # 205 (CSS пикселей)

# Расчёт frame coords (при корректном displayedContentRect):
frame_x = cr["x"] + display_click_x * (cr["width"] / cr["width"])   # 0 + 256 = 256
frame_y = cr["y"] + display_click_y * (cr["height"] / cr["height"])  # 115 + 205 = 320

print("frame_x:", frame_x, "frame_y:", frame_y)  # 256, 320

# Расчёт с НАЧАЛЬНЫМ displayedContentRect {x:0, y:0, w:512, h:640}:
frame_y_wrong = 0 + display_click_y * (TARGET_HEIGHT / cr["height"])
# = 0 + 205 * (640 / 410) = 0 + 320 = 320  ← СЛУЧАЙНО СОВПАЛО!

# НО при клике в верхней части контента (display_y=50):
frame_y_correct = cr["y"] + 50 * (cr["height"] / cr["height"])  # 115 + 50 = 165
frame_y_stale   = 0        + 50 * (TARGET_HEIGHT / cr["height"]) # 0 + 78 = 78  ← НЕВЕРНО!
print("correct:", frame_y_correct, "stale:", frame_y_stale)  # 165 vs 78 → разница 87px!
```

**Вывод**: race condition проявляется по-разному в зависимости от того, в какой части
контентной области кликает пользователь. Точно посередине — случайно близко к правильному,
в верхней/нижней частях — сильное отклонение.
