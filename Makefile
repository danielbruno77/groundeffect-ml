# Reproduce everything from scratch (about 1-2 h of CFD on a 14-core laptop).
.PHONY: setup test lint smoke grid doe retry dataset train sweep wt mgn figures all serve

setup:      ; uv sync --extra cfd && docker pull opencfd/openfoam-default:2512
lint:       ; uv run ruff check . && uv run ruff format --check . && uv run mypy src
test:       ; uv run pytest --cov
smoke:      ; uv run groundeffect smoke
grid:       ; uv run groundeffect grid-study
doe:        ; uv run groundeffect run-doe --workers 12
retry:      ; uv run groundeffect retry
dataset:    ; uv run groundeffect build-dataset
train:      ; uv run groundeffect train
sweep:      ; uv run groundeffect sweep
wt:         ; uv run groundeffect wt-demo
mgn:        ; uv run --extra physicsnemo groundeffect train-mgn
figures:    ; uv run groundeffect figures && uv run groundeffect active-learning
serve:      ; uv run groundeffect serve
all: grid doe retry dataset train sweep wt figures
