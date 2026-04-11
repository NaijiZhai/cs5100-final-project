# Traffic Signal Control Project

Single-intersection traffic signal control with **DQN**, redesigned to follow a paper-like control structure:

- fixed 4-phase cycle with safety phases
- action adjusts cycle timing (not instant phase switching)
- decision every full signal cycle
- history-aware observation
- comparison against fixed-time baseline

## Project Modules

### `traffic_signal_env.py`
Gymnasium environment for a single intersection.

Key design:

- **4 phases**: `NS green -> NS yellow -> EW green -> EW yellow`
- **3 actions**:
  - `0`: favor NS (increase NS green, decrease EW green)
  - `1`: keep base cycle
  - `2`: favor EW (decrease NS green, increase EW green)
- **cycle-level stepping**: one `env.step(action)` applies one action to one full 4-phase cycle
- **history-aware state**: current queue/wait + recent arrival/departure windows + phase info + last actions
- **reward components**:
  - waiting time penalty
  - queue penalty
  - NS/EW imbalance penalty
  - throughput reward
  - action-change penalty

### `agent.py`
DQN components:

- replay buffer (`ReplayBuffer`, backward alias `Memotable`)
- Q-network (`DQN`)
- agent wrapper (`Agent`)

### `traffic_dqn.py`
Training entry:

- builds paper-like env config
- epsilon-greedy data collection
- replay sampling + target network update
- saves training logs to `training_metrics.csv`
- saves model to `dqn_model.pth` and `dqn_checkpoint.pth`

### `baseline.py`
Evaluation policies:

- `FixedTimePolicy`: fixed static action (default keep / action `1`)
- `DemandAwareFixedTimePolicy`: fixed action chosen from demand (`arrival_prob`)
- `StaticActionPolicy`: fixed action baseline (`0`, `1`, or `2`)
- `RandomPolicy`: random action
- `QueueBasedPolicy`: favor NS/EW by queue imbalance with deadband
- `DQNPolicy`: wraps trained DQN model

### `evaluate.py`
Runs policy comparison over multiple episodes and reports averages:

- reward / queue / wait / imbalance
- throughput (`avg_departed`, `avg_departed_per_sec`)
- switch count (`avg_switch_count`)
- final queue / final wait (`avg_final_queue`, `avg_final_wait`)

Saves results to CSV:

- `eval_summary.csv` — aggregated metrics per policy
- `eval_raw.csv` — per-episode raw values per policy and seed
- `eval_queue_evolution.csv` — queue length per decision step (DQN and Fixed-Time)

Also supports loading either:

- plain `state_dict` (`dqn_model.pth`)
- metadata checkpoint (`dqn_checkpoint.pth`) with env/agent config

### `run_ablations.py`
Runs reproducible ablation sets:

- training duration (`600` vs `1800` episodes)
- model size (`hidden_dim 64` vs `128`)
- traffic difficulty (symmetric vs asymmetric peaky demand)

Outputs are saved under `results/ablations/...` with per-run `evaluation.json` and aggregated `summary.csv`.
The summary reports gains against multiple fixed-time baselines, including tuned static fixed-time (`reward_gain_vs_tuned`, `wait_reduction_vs_tuned`).

### `plot_training.py`
Plots detailed 6-panel training diagnostics from `training_metrics.csv` (reward, epsilon, queue, wait, imbalance, throughput).

### `plot_results.py`
Generates publication-ready result plots:

- training curves (reward and wait vs episode)
- bar chart comparison across all policies
- queue evolution over time (DQN vs Fixed-Time with mean ± std)

## Environment Interface

### Action Space
`Discrete(3)`:

- `0`: favor NS
- `1`: keep
- `2`: favor EW

### Observation Space
Flattened vector:

- 4 queue lengths
- 4 cumulative waits
- `K * 4` arrivals history
- `K * 4` departures history
- phase id
- phase remaining ratio
- last action
- previous action

When `normalize_state=True`, values are clipped/scaled to `[0, 1]`.

Note:

- this environment makes decisions at cycle boundaries (end of each full 4-phase cycle)
- therefore `phase id` and `phase remaining ratio` usually carry limited information (near-constant at decision time)
- the most informative features are queue/wait values, flow histories, and recent action history

### Episode Semantics

- `max_decisions` counts **decision steps**
- each decision step executes one complete cycle:
  - `NS green -> NS yellow -> EW green -> EW yellow`
- total simulated seconds in one episode are the sum of executed cycle durations
- `control_interval` is retained only for backward compatibility and is not used in cycle-level stepping

## Main Hyperparameters

`TrafficSignalEnv` useful args:

- `max_decisions` (decision horizon)
- `base_cycle` (`ns_green`, `ns_yellow`, `ew_green`, `ew_yellow`)
- `green_delta` (timing adjustment magnitude)
- `min_green`
- `history_windows`
- `arrival_prob`, `demand_variation`, `demand_period`
- reward weights:
  - `reward_wait_weight`
  - `reward_queue_weight`
  - `reward_imbalance_weight`
  - `reward_throughput_weight`
  - `reward_switch_weight`

## How to Run

### 0) Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (lighter install):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1) Train

```bash
python traffic_dqn.py
```

Outputs:

- `dqn_model.pth`
- `dqn_checkpoint.pth`
- `training_metrics.csv`

Useful tuning examples:

```bash
# longer training
python traffic_dqn.py --episodes 1800 --max-decisions 300 --epsilon-decay 450000

# larger model
python traffic_dqn.py --episodes 1200 --max-decisions 300 --hidden-dim 128 --epsilon-decay 360000
```

### 2) Evaluate

```bash
python evaluate.py
```

Compares:

- Fixed-Time (Keep)
- Fixed-Time (Demand-Aware)
- Fixed-Time (Tuned-Static via searching static actions `0/1/2`)
- Fixed-Time (Static a=0/1/2)
- Random
- Queue-Based
- DQN

Outputs:

- `eval_summary.csv` — avg reward, std, avg wait, avg queue per policy
- `eval_raw.csv` — per-episode raw values for each policy and seed
- `eval_queue_evolution.csv` — queue length per decision step (DQN and Fixed-Time)

You can force evaluation on custom env params or specific model file:

```bash
python evaluate.py --model-path dqn_checkpoint.pth --n-episodes 100
python evaluate.py --model-path dqn_model.pth --hidden-dim 128 --ignore-checkpoint-env
```

### 3) Plot results

```bash
python plot_results.py
```

Generates:

- `training_curves.png` — reward and wait vs episode during training
- `policy_comparison.png` — bar charts comparing all policies
- `queue_evolution.png` — DQN vs Fixed-Time queue over decision steps

The older `plot_training.py` is still available for detailed 6-panel training diagnostics.

### 4) Run Ablations

```bash
# run all 3 groups across 3 seeds
python run_ablations.py --group all --seeds 42,43,44 --eval-episodes 50

# run only model-size ablation
python run_ablations.py --group model_size --seeds 42,43,44
```

`results/ablations/summary.csv` includes:

- DQN mean reward/wait
- Fixed-Time Keep / Demand-Aware / Tuned-Static means
- `reward_gain_vs_keep`, `reward_gain_vs_demand`, `reward_gain_vs_tuned`
- `wait_reduction_vs_keep`, `wait_reduction_vs_demand`, `wait_reduction_vs_tuned`
- `avg_tuned_action` (mean best static action index over seeds)

How to interpret `summary.csv`:

- prioritize `reward_gain_vs_tuned`:
  - `> 0`: DQN outperforms the strongest static fixed-time baseline
  - `< 0`: tuned fixed-time is better than DQN in that setting
- prioritize `wait_reduction_vs_tuned`:
  - `> 0`: DQN reduces average waiting time vs tuned fixed-time
  - `< 0`: DQN increases average waiting time vs tuned fixed-time
- use `*_vs_keep` and `*_vs_demand` as secondary context; the tuned static baseline is the main fairness check

## Notes

- This is still a **single-intersection** study environment (not multi-agent/multi-junction).
- The control structure is intentionally aligned with fixed-cycle timing-adjustment methods so it can be migrated to SUMO later with minimal redesign.
