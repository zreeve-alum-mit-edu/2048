# 2048 RL Project — Milestones

## Overview

This document defines the project milestones following a fail-fast approach. Each milestone validates a specific capability before moving forward.

**Phase 1 Goal:** Agent reaches 8192 tile at least once in 100 games
**Phase 2 Goal:** Approach theoretical maximum average score

---

## Milestone Summary

| # | Milestone | Description |
|---|-----------|-------------|
| 1 | Test suite creation | Comprehensive tests before implementation |
| 2 | Game environment | Passes all tests |
| 3 | DQN basic | Trains/evals with one-hot, one reward, no tuning |
| 4 | Input representations | All transformations working |
| 5 | Hyperparam tuning (DQN full) | Full DQN sweep: all reprs x both rewards x tuning |
| 6 | Training Orchestrator | Can launch/manage multiple runs |
| 7-9 | Tier 1 algorithms | REINFORCE, A2C, Double DQN (basic) |
| 10-15 | Tier 2 algorithms | PPO variants, Dueling, PER, n-step, Rainbow-lite (basic) |
| 16-21 | Tier 3 algorithms | A3C, ACER, IMPALA, SARSA variants (basic) |
| 22-25 | Tier 4 algorithms (optional) | C51, QR-DQN, MuZero, MCTS (basic) |
| 26 | Full experimental sweep | All algorithms x all reprs x both rewards x tuning |
| 27 | Analysis & report | Results, best configs, Phase 1/2 goal check |

---

## Detailed Milestones

### Milestone 1: Test Suite Creation

**Objective:** Comprehensive test suite for GameEnv written BEFORE implementation.

**Deliverables:**
- All test scenarios brainstormed and documented
- Test fixtures defined (board states, expected outputs)
- Tests written using deterministic spawn injection
- Timing harness for GPU verification
- Tests runnable (failing against stub/mock)

**Success Criteria:**
- All test categories covered (see high-level-architecture.md section 10.4)
- GPU timing thresholds calibrated for GH200

---

### Milestone 2: Game Environment

**Objective:** GPU-native 2048 environment passes all tests from M1.

**Deliverables:**
- GameEnv implementation with precomputed lookup tables
- All M1 tests passing
- GPU-bound verification passing (device + timing)

**Success Criteria:**
- 100% test pass rate
- No CPU processing during game steps

---

### Milestone 3: DQN Basic

**Objective:** End-to-end training loop validated with simplest configuration.

**Configuration:**
- Algorithm: DQN
- Representation: one-hot
- Reward: merge_reward (or spawn_reward - pick one)
- No hyperparameter tuning

**Deliverables:**
- DQN algorithm implementation in `algorithms/dqn/`
- Training runs to completion
- Evaluation harness (100 games, avg/max score)

**Success Criteria:**
- Training loop runs without errors
- Agent learns (score improves over random baseline)

---

### Milestone 4: Input Representations

**Objective:** All input representation transformations working.

**Deliverables:**
- `representations/` module with:
  - one-hot (pass-through)
  - embedding (learned)
  - CNN (configurable kernels)
- DQN verified working with each representation

**Success Criteria:**
- All representations integrate with DQN
- CNN supports rectangular kernels and Inception-style multi-kernel

---

### Milestone 5: Hyperparameter Tuning (DQN Full)

**Objective:** Full Optuna pipeline validated on DQN.

**Configuration:**
- Algorithm: DQN
- Representations: ALL (one-hot, embedding, CNN variants)
- Rewards: BOTH (merge_reward, spawn_reward)
- Hyperparameter tuning: FULL

**Deliverables:**
- Optuna integration complete
- SQLite study storage working
- MedianPruner configured
- Full DQN sweep completed

**Success Criteria:**
- Studies run to completion for all (DQN x repr x reward) combos
- Best hyperparameters identified per combo
- Pruning working (bad trials killed early)

---

### Milestone 6: Training Orchestrator

**Objective:** Orchestrator can launch and manage multiple training runs.

**Deliverables:**
- Orchestrator implementation
- Config-driven experiment launching
- Parallel run management
- Metrics collection and aggregation
- Report generation

**Success Criteria:**
- Can launch multiple algorithm trainings
- Results collected and comparable
- Report shows comparative metrics

---

### Milestones 7-9: Tier 1 Algorithms (Basic)

Each algorithm trained with: one-hot, one reward, no tuning.

| # | Algorithm | Notes |
|---|-----------|-------|
| 7 | REINFORCE | Vanilla policy gradient, sanity check |
| 8 | A2C | Synchronous advantage actor-critic |
| 9 | Double DQN | DQN with target network improvement |

**Success Criteria per algorithm:**
- Training runs without errors
- Agent learns (improves over random)

---

### Milestones 10-15: Tier 2 Algorithms (Basic)

Each algorithm trained with: one-hot, one reward, no tuning.

| # | Algorithm | Notes |
|---|-----------|-------|
| 10 | PPO+GAE(lambda) | Headline method with generalized advantage |
| 11 | PPO+value clip | PPO variant with value function clipping |
| 12 | Dueling DQN | Separate value and advantage streams |
| 13 | PER DQN | Prioritized experience replay |
| 14 | n-step DQN | Multi-step returns |
| 15 | Rainbow-lite | Double + Dueling + PER + n-step |

**Success Criteria per algorithm:**
- Training runs without errors
- Agent learns (improves over random)

---

### Milestones 16-21: Tier 3 Algorithms (Basic)

Each algorithm trained with: one-hot, one reward, no tuning.

| # | Algorithm | Notes |
|---|-----------|-------|
| 16 | A3C | Async actor-critic (after A2C verified) |
| 17 | ACER | Actor-critic with experience replay |
| 18 | IMPALA/V-trace | Off-policy actor-learner |
| 19 | SARSA | On-policy TD control |
| 20 | Expected SARSA | Variance reduction |
| 21 | Q(lambda)/SARSA(lambda) | Eligibility traces |

**Success Criteria per algorithm:**
- Training runs without errors
- Agent learns (improves over random)

---

### Milestones 22-25: Tier 4 Algorithms (Optional/Stretch)

Each algorithm trained with: one-hot, one reward, no tuning.

| # | Algorithm | Notes |
|---|-----------|-------|
| 22 | C51 | Distributional DQN |
| 23 | QR-DQN | Quantile regression DQN |
| 24 | MuZero-style | Model-based planning (heavy) |
| 25 | MCTS+learned | Monte Carlo tree search with learned value/policy |

**Success Criteria per algorithm:**
- Training runs without errors
- Agent learns (improves over random)

---

### Milestone 26: Full Experimental Sweep

**Objective:** All algorithms x all representations x both rewards x full hyperparameter tuning.

**Scope:**
- Every implemented algorithm
- All representation types (one-hot, embedding, CNN variants)
- Both reward types (merge_reward, spawn_reward)
- Full Optuna tuning per (algo x repr x reward) combo

**Deliverables:**
- All studies completed
- Best hyperparameters per combo
- Comparative metrics across all configurations

**Success Criteria:**
- All studies run to completion
- Results collected and aggregated

---

### Milestone 27: Analysis & Report

**Objective:** Analyze results, identify best configurations, check Phase 1/2 goals.

**Deliverables:**
- Final report with:
  - Best performing (algo x repr x reward x hyperparams) configs
  - Phase 1 check: Did any config hit 8192 in 100 games?
  - Phase 2 metrics: Average scores, comparison to theoretical max
  - Insights: What worked, what didn't, why
- Visualizations: Learning curves, score distributions, comparison charts

**Success Criteria:**
- Phase 1 goal met (at least one config hits 8192)
- Clear ranking of approaches
- Actionable insights documented
