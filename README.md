# gym-rock-paper-scissors
Rock Paper scissors environment for OpenAI Gym environment

Rock-paper-scissors environment is an implementation of the **repeated game** of rock-paper-scissors. Where the agents repeatedly play the **normal form** game of rock paper scissors.

## Action space

The action set is common to all agents, and it contains three elements: `[ROCK, PAPER, SCISSORS]`.

## State space

The normal form version of rock paper scissors does not have a state representation *per se*. However we can represent the state of a repeated game by keeping track of the last actions taken by each player for the last `n` iterations of the game. Let `n` be an environment parameter, and let *(a<sup>1</sup><sub>t</sub>, a<sup>2</sup><sub>t</sub>)* be the action pair for both player 1 and 2 at timestep *t*. The state representation becomes *[(a<sup>1</sup><sub>0</sub>, a<sup>2</sup><sub>0</sub>), (a<sup>1</sup><sub>1</sub>, a<sup>2</sup><sub>1</sub>), ..., (a<sup>1</sup><sub>n</sub>, a<sup>2</sup><sub>n</sub>)]*

## Reward function

Follows the classical rules of rock paper scissors. Rock beats scissors, scissors beats paper, paper beats rock. If both players take the same action, they both get get a reward of `0`.

## Installation

```bash
cd gym-rock-paper-scissors
pip install -e .
```
