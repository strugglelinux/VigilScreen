#!/bin/zsh
cd "${0:A:h}"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python main.py
elif [[ -x ../pytorch_env/bin/python ]]; then
  exec ../pytorch_env/bin/python main.py
else
  exec python3 main.py
fi
