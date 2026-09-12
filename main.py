# ELSADIGAI

## Learning Intelligence System

ELSADIGAI is an experimental learning-intelligence system designed to study how an agent can learn from experience, store previous experiences, evaluate actions, and improve its future decisions.

## Core Idea

The system follows a simple learning cycle:

Perception → Action → Reward → Memory → Learning → Better Decision

## Main Components

- `backend.py` — learning engine and memory
- `frontend.py` — graphical user interface
- `main.py` — application launcher

## Learning Test

ELSADIGAI includes an experimental benchmark:

1. Before learning
2. Training
3. After learning
4. Generalization test

The benchmark compares:

- Average reward
- Best-action rate
- Improvement after training

## Status

**Experimental / Research Prototype**

The project does not claim to be artificial general intelligence or superintelligence. Its purpose is to experimentally investigate learning and adaptive decision-making.

## Requirements

Python 3.x

The basic prototype uses Python standard-library components.

## Running

```bash
python main.py