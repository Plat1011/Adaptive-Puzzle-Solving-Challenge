# Adaptive Puzzle Solving Challenge

Соревнование Яндекса. Для каждой скрытой головоломки на стороне проверяющей
системы запускается train.py (50 минут CPU), потом solve.py (25 минут CPU
на 1000 инстансов: 300 публичных + 700 приватных). Всё на 8 ядрах, 32 ГБ
RAM, без GPU. Образ Docker меньше 20 ГБ.

Головоломка задана `gym.py` с действиями `SWAP`, `ROTATE`, `TOGGLE`,
`PERMUTE`. Метрика на один инстанс: `baseline_length / participant_moves`,
ограничено 2.0; невалидное или непринятое решение даёт 0.

## Структура

```
solution/             # код, который идёт в архив сабмита
  train.py            # запускается на каждой скрытой игре отдельно (50 мин)
  solve.py            # потом запускается с input_states.jsonl (25 мин)
  worker.py           # subprocess-воркер для параллельного solve
  common.py           # state_key, токенизация, backward walks
  detect.py           # пробинг среды: типы действий, ветвление, бинарность
  gf2_solver.py       # решение TOGGLE-only головоломок через систему GF(2)
  compact_table.py    # mmap-таблица BFS-расстояний от цели
  goal_table.py       # старая dict-based версия таблицы
  bidir_bfs.py        # двунаправленный BFS (фолбэк когда нет goal-table)
  beam.py             # beam search с goal-table в качестве цели
  astar.py            # A* с произвольной эвристикой
  heuristics.py       # Manhattan-подобная эвристика на основе encode_state
  model.py            # ValueNet от организаторов (не используется)
  Dockerfile          # python:3.11-slim + numpy + torch CPU

baseline/             # бейзлайн организаторов как есть
scripts/              # локальные smoke-тесты, test_local.sh
docs/methods.csv      # журнал подходов и результатов
```

## Как работает

`solve.py` загружает то, что оставил `train.py`, и для каждого инстанса
пробует несколько стратегий по очереди. Первая, которая нашла валидное
решение, побеждает.

1. GF(2) солвер. Если пробинг сказал что все действия типа TOGGLE,
   инволютивные, а ячейки бинарные, строится булева матрица "действие
   переключает каких ячеек" и задача сводится к решению Ax = b над GF(2)
   обычным гауссом. Оптимально по числу действий (с точностью до
   перестановки) и работает за миллисекунды. Закрывает Lights Out и
   родственные.

2. Forward BFS до goal-table. Пока шёл train, мы прогнали BFS от
   решённого состояния и сохранили шар достижимых состояний с их
   глубинами. На этапе solve запускаем BFS от инстанса; как только
   попали в шар, склеиваем forward-путь с реконструкцией из таблицы.
   Если попали, путь оптимальный.

3. Beam search до goal-table. Если простой BFS не дотянулся, идём
   beam-ом с Manhattan-эвристикой. Ширина адаптивная по среднему
   ветвлению, плюс iterative widening: пробуем `beam_w`, потом
   `2 * beam_w`, потом `4 * beam_w` пока остаётся время.

4. Bidirectional BFS на случай если goal-table не построилась.

5. Weighted A* (h_weight = 1.5) как финальный фолбэк.

## Goal-table в компактном mmap-формате

Самый полезный апгрейд. Вместо pickled Python `dict {state_key:
(parent_key, action, depth)}` (по ~150 байт на запись с учётом оверхеда
объектов) храним четыре numpy-массива:

```
gt_keys.npy     uint64   отсортированные blake2b-хеши состояний
gt_parents.npy  uint64   хеши родителей
gt_actions.npy  uint16   индексы в gt_vocab.json
gt_depths.npy   uint16   глубины от цели
```

Получается 20 байт на запись вместо 150, плюс при `np.load(...,
mmap_mode='r')` на Linux все воркеры делят одну копию через page cache.
То есть таблица грузится один раз независимо от количества воркеров.
Лукап через `np.searchsorted` по отсортированным хешам, в районе 50 нс.

С таким форматом и 32 ГБ RAM реально держать в таблице 10-15 миллионов
состояний.

## Параллелизм через subprocess

Сначала был `multiprocessing.Pool` с fork. На контестном Docker оно
падало на всех тестах сразу, локально не воспроизводилось. Подозрение на
fork-after-torch-import или seccomp в их образе, разбираться было
некогда.

В итоге сделано через `subprocess.Popen`: главный `solve.py` делит
инстансы на 8 чанков, спавнит 8 независимых Python-процессов с
`worker.py`, ждёт их, собирает выходные CSV в итоговый. Никаких проблем
с наследованием состояния интерпретатора, OMP и прочей магии. Воркеры
делят только mmap-нутую таблицу.

## Локальный запуск

```
TRAIN_TIME_LIMIT=120 SOLVE_TIME_LIMIT=150 SOLVE_WORKERS=8 \
    bash scripts/test_local.sh game_15_2d 100
```

Скрипт делает `generate_states.py` -> `train.py` -> `solve.py` ->
`check.py` в `_work/<env>/`.

## Журнал

Полная история в [docs/methods.csv](docs/methods.csv). Короткая сводка:

```
v1   GF(2) + bidirectional BFS                    local mean 1.05
v2   axis-scaled Manhattan heuristic               local mean 1.34
v3   goal-table BFS, faster state_key, beam        local mean 1.65   LB 68.5
v4   multiprocessing.Pool (fork)                                     LB 0 (crash 13/13)
v5   откат на single-thread, defensive try/except  local mean 1.65
v6   параллелизм через subprocess.Popen            local mean 1.66   LB 77
v7   bumped beam_w to 256                          local mean 1.66   LB 75
v8   compact mmap goal table, cap 15M states       local mean 1.69
```

Замеры local mean - среднее по 3 открытым играм при 1.5 с на инстанс и
2-минутном train. На контесте train в 25 раз больше, так что таблица
получается жирнее и реальный score должен быть выше.

### Что сработало

GF(2) для TOGGLE-only головоломок. Никакого поиска не нужно вообще, всё
решается линейной системой. На этом классе получаем максимум сразу.

Goal-table backward BFS. Пока есть время train, BFS от цели набирает
несколько миллионов состояний с известными расстояниями. На этапе
solve forward-BFS / beam от инстанса попадает в шар - и сразу
оптимальный путь.

Компактный mmap-формат. Старый pickled dict не масштабировался: воркеры
грузили его независимо, и 4M записей умножались на 8 копий, что
вылетало по OOM на контесте. С mmap arrays одна копия делится на всех,
размер таблицы стал ограничен build-time, а не run-time памятью.

Subprocess-параллелизм. Простой и стабильный. Даёт 4-5x по wall-time
без подводных камней multiprocessing.

Адаптивная ширина beam по среднему ветвлению. У cylinder ветвление 13+,
там узкий beam успевает больше шагов и побеждает; у game_15 ветвление
3-4, широкий beam лучше.

### Что не сработало

`multiprocessing.Pool` с fork. Локально нормально, на контесте crash.

NN-эвристика в beam-search. Идея обучить ValueNet на (state, depth)
парах из BFS-таблицы и оценивать кандидатов в beam через NN. Бутылочное
горлышко - `encode_tokens` на каждое состояние; даже с батч-вызовом
torch выходит дороже, чем выигрывает в качестве. Score падал на 30%
именно из-за того, что меньше инстансов успеваем обработать.

Слишком широкий beam (512+). Каждый шаг линеен по `beam_w *
branching_factor`, а число успешных раундов quadratically падает.
Оптимум 64-256.

Бамп goal-table cap до 4M без mmap. На сабмите словили ML
(превышение памяти) на 2 тестах из 13 - воркеры грузили pickle dict
независимо и умножали память.

## Что планирую попробовать

IDA* с linear-conflict для глубоких scramble-ов 15-puzzle-стиля -
текущий beam там оставляет path-ы на 60% длиннее оптимума.

Pattern database для SWAP-only с фиксированной геометрией.

DeepCubeA / Q*-style NN если доберусь до батчевого encode без
`env.encode_state` на каждый вызов.

## Метрика

```
per_instance = min(2.0, baseline_length / participant_moves)
per_puzzle   = mean(per_instance)
total        = aggregate by judge over 13 hidden puzzles
```

Невалидное или нерешённое = 0.

## Сборка архива

```
cd solution && zip -r ../submission.zip *.py Dockerfile
```

В корне архива должны быть `train.py`, `solve.py`, `Dockerfile`,
остальные `.py` рядом.
