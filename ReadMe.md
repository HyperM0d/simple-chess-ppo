# ppo chess

a simple ppo framework for learning chess through self-play

## structure

```
ppo_chess/
  encoder.py    - board to neural net input
  network.py    - ppo policy + value network
  buffer.py     - trajectory storage
  env.py        - chess game wrapper
  agent.py      - ppo agent + training functions
  main.py       - run training
  __init__.py   - exports
```

## install

```bash
pip install torch numpy chess
```

## run

```bash
python -m ppo_chess.main
```

## usage

```python
from ppo_chess import ppo_agent, self_play, evaluate

agent = ppo_agent(lr=1e-4)

self_play(agent, num_games=100)

results = evaluate(agent, num_games=20)
```

## TODO
- [ ] Model saving
- [ ] Model loading
- [ ] Model training
