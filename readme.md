# Traffic Signal Control Project

This is a simple traffic signal control project based on **DQN (Deep Q-Network)**.  
The goal is to train an agent to switch traffic light phases in order to reduce vehicle queues and waiting time.

The main DQN implementation is based on the core idea from:

> **Playing Atari with Deep Reinforcement Learning**  
> <https://arxiv.org/pdf/1312.5602>

In other words, the code follows the standard DQN pipeline, including:

- experience replay
- target network
- epsilon-greedy exploration
- neural network approximation of Q-values

---

## Project Modules

### `traffic_signal_env.py`
This file defines the **traffic signal environment**.

Main responsibilities:

- define the observation space
- define the action space
- simulate vehicle arrivals and departures
- compute rewards
- return step statistics

You can think of this file as the **environment that the agent interacts with during training and evaluation**.

---

### `agent.py`
This file contains the main DQN-related components.

#### `Memotable`
This is the replay buffer. It stores transitions of the form:

- current state `s`
- action `a`
- reward `r`
- terminal flag `done`
- truncation flag `trunc`
- next state `s_new`

During training, batches are randomly sampled from this buffer.

#### `DQN`
This is the Q-network, implemented as a simple MLP.  
It takes the environment state as input and outputs Q-values for all actions.

#### `Agent`
This class groups together:

- the online network
- the target network
- the replay buffer
- the optimizer
- other training-related hyperparameters

---

### `traffic_dqn.py`
This is the **main training script** for DQN.

Main steps:

- create the environment
- create the agent
- interact with the environment in each episode
- choose actions with epsilon-greedy
- store transitions in the replay buffer
- sample batches and train the online network
- periodically update the target network
- print training statistics such as reward and throughput

This file is basically the **training entry point**.

---

### `baseline.py`
This file defines several baseline policies for comparison.

#### `FixedTimePolicy`
Switches the traffic light every fixed number of steps.

#### `RandomPolicy`
Chooses actions randomly.

#### `DQNPolicy`
Wraps a trained DQN agent into a policy interface so it can be evaluated in the same way as the baselines.

The purpose of this file is to provide **simple reference strategies** for comparison.

---

### `evaluate.py`
This file is used to **evaluate different policies**.

It compares:

- Fixed-Time Policy
- Random Policy
- DQN Policy

It reports metrics such as:

- average reward
- average departed vehicles
- average final queue length
- average final waiting time
- average switch count

This file is mainly for experiment comparison.

---

## DQN Workflow

The DQN implementation follows the basic procedure from  
**Playing Atari with Deep Reinforcement Learning**.


## Environment Parameters

The `TrafficSignalEnv` currently supports the following main parameters:

### `max_steps=1000`
Maximum number of steps in one episode.

- the episode is truncated when this limit is reached
- controls the length of each training/evaluation run

---

### `arrival_prob=(0.3, 0.3, 0.3, 0.3)`
Arrival probabilities for the four incoming lanes.

- there are 4 lanes in total
- at each step, each lane may receive a new vehicle according to its probability

Example:

```python
arrival_prob=(0.5, 0.2, 0.5, 0.2)
```


This means some directions have heavier traffic flow than others.

---

### `depart_capacity=2`
Maximum number of vehicles that can leave each green lane in one step.

- larger values mean higher traffic throughput per step
- directly affects congestion and flow efficiency

---

### `min_green_steps=3`
Minimum number of steps a phase must stay green before switching is allowed.

- prevents the signal from switching too frequently
- introduces a simple control constraint

---

### `max_queue=50`
Upper bound used when normalizing queue lengths.

If state normalization is enabled, queue lengths are divided by this value and clipped at `1.0`.

---

### `max_wait=500`
Upper bound used when normalizing waiting times.

If state normalization is enabled, waiting times are divided by this value and clipped at `1.0`.

---

### `normalize_state=True`
Whether to normalize the state.

- `True`: returns normalized state values
- `False`: returns raw state values

In general, normalized states are more suitable for neural network training.

---

## State, Action, and Reward

### State
The state contains 9 values:

- 4 lane queue lengths
- 4 lane accumulated waiting times
- 1 current phase indicator

---

### Action
The action space has 2 discrete actions:

- `0`: north-south green
- `1`: west-east green

---

### Reward
The current reward is designed to:

- penalize total queue length
- penalize total waiting time
- penalize phase switching

So the agent is encouraged to:

- reduce congestion
- reduce waiting
- avoid unnecessary switching

---

## How to Run

### Train the DQN agent

```shell script
python traffic_dqn.py
```


### Evaluate different policies

```shell script
python evaluate.py
```


---

## Current Scope

This project is currently a **basic DQN traffic signal control framework**.  
The main focus is:

- a working environment
- a complete DQN training pipeline
- comparison against simple baselines

Possible future improvements include:

- reward design
- richer state features
- better network architecture
- hyperparameter tuning
- more stable training methods

---

