# ELSADIGAI

## Learning Intelligence System

ELSADIGAI is an experimental learning-intelligence system designed to study how an agent can learn from experience, store previous experiences, evaluate actions, and improve its future decisions.

## Core Idea

The system follows a simple learning cycle:

**Perception → Action → Reward → Memory → Learning → Better Decision**

## Main Components

- `backend.py` — learning engine and memory
- `frontend.py` — graphical user interface
- `main.py` — application launcher

## Learning Test

ELSADIGAI includes an experimental benchmark:

1. **Before learning** — Random behavior baseline
2. **Training** — 500 episodes of learning
3. **After learning** — Testing improved behavior
4. **Generalization test** — Testing on unseen states

The benchmark compares:

- Average reward
- Best-action rate
- Improvement after training

## Architecture

### Memory System
- Stores experiences in JSON format
- Keeps last 1000 experiences
- Auto-saves to disk

### Learning Agent
- Q-learning style value estimation
- Epsilon-greedy action selection
- Incremental value updates

### Environment
- 8 training states (A-H)
- 4 generalization states (I-L)
- Hidden optimal actions per state

## Status

**Experimental / Research Prototype**

The project does not claim to be artificial general intelligence or superintelligence. Its purpose is to experimentally investigate learning and adaptive decision-making.

## Requirements

- Python 3.7+
- tkinter (usually included with Python)

## Running

```bash
python main.py
```

## Testing

```bash
python -m pytest tests/
```

## Files Structure

```
ELSADIGAI/
├── main.py              # Application entry point
├── frontend.py          # Tkinter UI
├── backend.py           # Learning engine
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── tests/
    └── test_backend.py # Unit tests
```

## License

MIT License

## Author

ELSADIGAI Project Team
