# Adaptive Puzzle Solving Challenge

Соревнование по разработке универсального алгоритма для решения обратимых дискретных
головоломок. Каждая головоломка задаётся `gym.py`-средой с единым интерфейсом.

## Условия

- 8 ядер CPU, без GPU.
- Один общий формат состояний (ячейки в 3D + типы содержимого) и действий
  (`SWAP`, `ROTATE`, `TOGGLE`, `PERMUTE`).
- На каждую hidden-головоломку отдельный запуск: `train.py` (≤ 50 мин)
  → `solve.py` (≤ 25 мин на 1000 примеров).
- Открытые примеры в `baseline/gym.py`: `game_15_2d`, `toggle_lights`, `cylinder_game`.
- Метрика на состояние: `baseline_length / participant_moves`, итог — среднее по набору.
  Неверное / пустое решение даёт 0.
- 40 сабмитов всего, по 10 в день.

## Структура

```
Adaptive Puzzle Solving Challenge/
├── baseline/        # организаторский бейзлайн (ValueNet + A*)
├── src/             # наш код
├── scripts/         # локальные эксперименты
├── docs/            # заметки, планы
├── README.md
└── requirements.txt
```

## Запуск бейзлайна

```bash
cd baseline
./test_local.sh toggle_lights 10
# или с укороченным временем
TRAIN_TIME_LIMIT=60 SOLVE_TIME_LIMIT=30 ./test_local.sh toggle_lights 2
```

## Идея бейзлайна

Маленькая `ValueNet` учится оценивать число шагов до решённого состояния
(на random-walk’ах из цели). В `solve.py` используется как эвристика в A*.
Сеть универсальна — работает через `env.encode_state` и `env.valid_actions`,
не зная конкретной игры.
