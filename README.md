# Adaptive Puzzle Solving Challenge

Соревнование Яндекса по разработке универсального алгоритма для решения обратимых
дискретных головоломок. Каждая головоломка задана `gym.py`-средой с единым
интерфейсом: состояние из дискретных ячеек, набор обратимых действий (`SWAP`,
`ROTATE`, `TOGGLE`, `PERMUTE`), известное целевое состояние. Задача — найти
короткую последовательность действий до цели.

Оценка лидерборда строится на скрытых головоломках, не известных заранее.
Метрика на состояние: `baseline_length / participant_moves` (cap 2.0), 0 за
невалидное / непринятое.

Ограничения:
- 50 минут CPU на `train.py` для каждой головоломки.
- 25 минут CPU на `solve.py` (1000 инстансов: 300 публичных + 700 приватных).
- 8 ядер CPU, 32 ГБ RAM. Без GPU.
- Размер Docker-образа < 20 ГБ.

## Структура

```
solution/             # код, который идёт в архив сабмита
├── train.py          # запускается на каждой скрытой игре отдельно (50 мин)
├── solve.py          # потом запускается с input_states.jsonl (25 мин)
├── worker.py         # subprocess-воркер для параллельного solve
├── common.py         # state_key, токенизация, backward walks
├── detect.py         # пробинг среды: типы действий, ветвление, бинарность
├── gf2_solver.py     # решение TOGGLE-only головоломок через систему GF(2)
├── compact_table.py  # компактная mmap-таблица BFS-расстояний от цели
├── goal_table.py     # старая (dict-based) версия таблицы, оставлена как фолбэк
├── bidir_bfs.py      # двунаправленный BFS (фолбэк когда нет goal-table)
├── beam.py           # beam search с компактной goal-table в качестве цели
├── astar.py          # A* с произвольной эвристикой (final fallback)
├── heuristics.py     # Manhattan-подобная эвристика на основе encode_state
├── model.py          # ValueNet (от организаторов, не используется сейчас)
└── Dockerfile        # python:3.11-slim + numpy + torch CPU

baseline/             # оригинальный бейзлайн организаторов (для справки)
scripts/              # локальные эксперименты (smoke-тесты, test_local.sh)
docs/                 # methods.csv — журнал подходов и результатов
```

## Архитектура решения

Адаптивный диспетчер. При запуске `solve.py` загружает артефакты, оставленные
`train.py`, и для каждого инстанса перебирает стратегии в порядке убывания
надёжности:

1. **GF(2) solver** — если head-пробинг говорит, что все действия — `TOGGLE`,
   инволютивные, ячейки бинарные. Строится булева матрица «действие → какие
   ячейки переключает», задача сводится к решению `Ax = b` над GF(2)
   гауссом. Гарантированно оптимально по числу действий (с точностью до
   перестановки), работает за миллисекунды. Закрывает все Lights-Out-подобные
   игры.

2. **Forward BFS до goal-table** — за время train построена «шар» состояний
   рядом с целью (BFS от solved). При поиске запускаем forward BFS от начального
   состояния, первое попадание в shar → склеиваем форвард-путь с реконструкцией
   из таблицы. Даёт оптимальный путь при попадании.

3. **Beam search до goal-table** — если простой BFS не успел дотянуться,
   beam-search с Manhattan-эвристикой на основе позиций из `encode_state`.
   Iterative widening: пробуем `beam_w`, потом `2*beam_w`, потом `4*beam_w`.
   `beam_w` адаптируется по среднему ветвлению (`profile.json`): мало действий
   на ход — широкий beam, много действий — узкий.

4. **Bidirectional BFS** — фолбэк когда goal-table не построилась.

5. **Weighted A*** — финальный фолбэк с эвристикой и пониженным admissibility
   (`h_weight = 1.5`).

### Goal-table в компактном mmap-формате

Главная находка. Вместо pickled Python `dict {state_key: (parent_key, action,
depth)}` (~150 байт/состояние с учётом оверхеда объектов) храним 4 numpy
массива:

- `gt_keys.npy` — отсортированные `uint64` хеши состояний (blake2b 8 байт)
- `gt_parents.npy` — `uint64` хеши родителей
- `gt_actions.npy` — `uint16` индексы в `gt_vocab.json`
- `gt_depths.npy` — `uint16` глубины

Итого ~20 байт/состояние, в 7-8 раз компактнее.

Worker'ы открывают файлы через `np.load(..., mmap_mode='r')`. Linux разделяет
mmap'нутые страницы между процессами через page cache: одна копия в памяти на
все 8 worker'ов. Лукап через `np.searchsorted` — ~50нс.

С этим форматом и 32 ГБ RAM реально вместить до 15-20 миллионов состояний в
таблицу.

### Параллелизм через subprocess

После неудачного эксперимента с `multiprocessing.Pool` (см. журнал), параллелизм
сделан через `subprocess.Popen`: главный `solve.py` делит инстансы на 8 чанков,
запускает 8 независимых Python-процессов с `worker.py`, ждёт, собирает CSV.
Никаких проблем с fork-after-import, OpenMP, наследованием состояния
интерпретатора. Worker'ы делят только mmap'нутую таблицу.

## Локальный запуск

```bash
TRAIN_TIME_LIMIT=120 SOLVE_TIME_LIMIT=150 SOLVE_WORKERS=8 \
    bash scripts/test_local.sh game_15_2d 100
```

## Журнал подходов

Полная история — в [docs/methods.csv](docs/methods.csv). Краткие итоги:

| Версия | Mean (local) | Контест | Что добавили |
|---|---:|---:|---|
| baseline (организаторы) | ~0.33 | — | A* с маленькой ValueNet |
| v1 | 1.05 | — | + GF(2), + bidirectional BFS |
| v2 | 1.34 | — | + Manhattan-эвристика (axis-scaled) |
| v3 | 1.65 | 68.5 | + goal-table BFS, faster state_key, beam search |
| v4 | — | 0 (crash) | попытка `multiprocessing.Pool` с fork |
| v5 | 1.65 | — | откат mp, defensive try/except |
| v6 | 1.66 | 77 | параллелизм через `subprocess.Popen` |
| v7 | 1.66 | 75 | bump beam_w, мелкий тюнинг (-2 балла) |
| v8 | 1.69 | — | compact mmap-таблица — главный апгрейд |

### Что сработало

- **GF(2) для TOGGLE-игр**. Lights-Out не требует поиска вообще — решается
  гауссом. Дало почти максимум на этом классе с первого подхода.
- **Goal-table BFS**. Backward BFS от цели на 5+ млн состояний + forward
  BFS/beam от начала → оптимальные пути там, где они помещаются в шар.
- **Compact mmap-формат**. Развязал руки по размеру таблицы: было 2M на 700МБ
  pickle, стало 10M+ на 100МБ mmap, разделяемой между всеми worker'ами.
- **Subprocess-параллелизм**. Простой и устойчивый. Дал реальное 4-5×
  ускорение wall-time без проблем с fork/spawn.
- **Адаптивный `beam_w`** по среднему ветвлению из профиля среды.

### Что не сработало

- **`multiprocessing.Pool` с fork** — на контестном Docker всё крашнулось на
  всех 13 тестах. Без воспроизведения локально — скорее всего конфликт
  fork-after-torch-import или ограничения seccomp. Откатились на subprocess.
- **NN-эвристика в beam-search**. Кажущаяся неплохой идеей — обучить ValueNet
  на (state, true_depth) парах из BFS-таблицы, использовать как scoring в beam.
  На практике per-state `encode_tokens` слишком медленный → каждая итерация
  beam в разы дольше → меньше инстансов успеваем → score падает.
- **Слишком большой `beam_w`** (512+). Каждый шаг beam'а O(beam_w * branching)
  по времени; широкий beam = меньше итераций успеваем = меньше глубоких
  инстансов решаем. Оптимум 64-256 в зависимости от ветвления.
- **Простой бамп goal-table cap до 4M без mmap**. Воркеры грузили pickle dict
  независимо → 4ГБ × 8 = OOM на контесте (ML на 2 тестах из 13).

### Что осталось попробовать

- IDA\* с linear-conflict для глубоких 15-puzzle-like scramble'ов.
- Pattern-database на разбиении состояний (для SWAP-игр).
- DeepCubeA-стиль: NN, обученная value-iteration'ом на reverse walks, в
  паре с BWAS (Batch-Weighted A\* Search).

## Метрика

```
score_per_instance = min(2.0, baseline_length / participant_moves)
score_per_puzzle   = mean(score_per_instance)
total              = aggregate by contest system over 13 hidden puzzles
```

Невалидное или нерешённое — 0 за инстанс.

## Сборка submission.zip

```bash
cd solution && zip -r ../submission.zip *.py Dockerfile
```

Архив должен содержать как минимум `train.py`, `solve.py`, `Dockerfile` в
корне.
