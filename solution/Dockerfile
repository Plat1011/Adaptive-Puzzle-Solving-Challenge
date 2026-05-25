FROM python:3.11-slim

RUN pip install --no-cache-dir numpy tqdm \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY train.py solve.py worker.py common.py model.py detect.py bidir_bfs.py astar.py gf2_solver.py heuristics.py goal_table.py compact_table.py beam.py nn_heuristic.py /opt/participant/

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/opt/participant
