# 2048 RL Project - High-Level Architecture

## 1. Project Goal

Train a reinforcement learning agent that plays 2048 at a high level.

**Success Criteria (Phased):**

| Phase | Criterion | Description |
|-------|-----------|-------------|
| 1 | 8192 Reach | Agent reaches 8192 tile at least once in 100 games |
| 2 | Score Optimization | Approach theoretical maximum average score |

---

## 2. High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Orchestrator                      │
│  (kicks off all trainings, gathers metrics, final report)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │         RL Algorithm Modules            │
        │  (each has own training & eval runner)  │
        └───────────────────┬─────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────┐
        │         Standardized Interface          │
        ├─────────────────────────────────────────┤
        │           Game Environment              │
        │  (GPU-native tensors, N parallel games) │
        └─────────────────────────────────────────┘
```

| Component | Responsibility |
|-----------|----------------|
| **Training Orchestrator** | Single entry point. Launches algorithm trainings, collects metrics/logs, produces final comparison report. |
| **RL Algorithm Modules** | Self-contained per algorithm. Each has its own training loop and evaluation runner. |
| **Game Environment** | GPU-native 2048 using PyTorch tensors. Configurable N parallel games. |

---

## 3. Standardized Interface (Env - Agent)

**Env to Agent:**

| Data | Shape | Notes |
|------|-------|-------|
| Game state | `(N, 16, 17)` | One-hot: 16 positions x 17 values (0=empty, 1-16=2^1-2^16) |
| Valid actions mask | `(N, 4)` | Legal moves |
| Done flags | `(N,)` | Game ended |
| Merge reward | `(N,)` | Sum of merged tiles |
| Spawn reward | `(N,)` | Value of spawned tile (2 or 4) |
| Reset states | `(N, 16, 17)` | Fresh boards for games where done=True |

**Agent to Env:**

| Data | Shape | Notes |
|------|-------|-------|
| Actions | `(N,)` | int 0-3 (up/down/left/right) |

---

## 4. Technical Constraints

| Constraint | Requirement |
|------------|-------------|
| **Framework** | PyTorch |
| **Hardware** | NVIDIA GH200 (ARM + Hopper GPU) |
| **ARM Compatibility** | All dependencies MUST support aarch64 |
| **Parallelism** | Design for multi-env parallel training |

---

## 5. Game Environment Internals

### 5.1 Precomputed Lookup Tables

Generated once at startup, stored on GPU.

| Table | Shape | Purpose |
|-------|-------|---------|
| Line transition | `(17, 17, 17, 17, 4, 17)` | Result of left move on any 4-tile line |
| Valid move | `(17, 17, 17, 17)` bool | Is left move valid (line changes) |
| Score delta | `(17, 17, 17, 17)` int | Points earned from merges |

### 5.2 Tile Spawning

| Aspect | Value |
|--------|-------|
| Values | 2 (90%) or 4 (10%) |
| Location | Random empty cell |
| When | After each valid move |

### 5.3 Reward Signal

Environment returns BOTH reward types; algorithm config selects which to use:

| Type | Signal | Source |
|------|--------|--------|
| `merge_reward` | Sum of merged tiles | From precomputed score delta table |
| `spawn_reward` | Value of spawned tile | From tile spawn step |

### 5.4 Episode Boundary Handling (CRITICAL)

**Invariants - MUST be enforced:**
- When `done=True`, `next_state` is the terminal state, NOT the new game's start
- Experience replay buffers MUST NOT contain cross-episode transitions
- `reset_states` returned separately for games that ended

**Testing requirements:**
- Unit tests verify replay buffer never contains cross-episode transitions
- Trajectory collectors tested for boundary respect
- Stress test with synthetic instant-game-over scenarios

---

## 6. Open Items (Future Design Iterations)

- Training Orchestrator details (config format, parallelism, reporting)
- Directory structure (beyond algorithms/ and representations/)

---

## 7. RL Algorithm Module Structure

### 7.1 Location & Convention

Each algorithm is fully self-contained in its own folder:

```
algorithms/<name>/
├── run.py           # Required entry point
├── model.py         # Network architecture
├── config.yaml      # Algorithm-specific hyperparameters
└── ...              # Any other algorithm-specific code
```

### 7.2 Interface

Each algorithm's `run.py` MUST implement:

```python
def train(env_factory, config_path) -> TrainingResult:
    """
    Args:
        env_factory: callable that returns a new GameEnv instance
        config_path: path to algorithm-specific config file
    Returns:
        TrainingResult with checkpoint paths and training metrics
    """
    ...

def evaluate(env_factory, checkpoint_path, num_games) -> EvalResult:
    """
    Args:
        env_factory: callable that returns a new GameEnv instance
        checkpoint_path: path to saved model
        num_games: how many games to run
    Returns:
        EvalResult with scores
    """
    ...
```

**Standardized result types:**

```python
@dataclass
class TrainingResult:
    checkpoints: list[str]      # Saved checkpoint paths
    metrics: dict               # Algorithm-specific (free-form)

@dataclass
class EvalResult:
    scores: list[int]           # Final score per game
    avg_score: float            # Average across all games
    max_score: int              # Best score achieved
```

### 7.3 Algorithms by Tier

| Tier | Algorithms | Notes |
|------|------------|-------|
| **1 — Core** | REINFORCE, A2C, DQN, Double DQN | Implement first, baselines |
| **2 — Primary** | PPO+GAE(λ), PPO+value clip, Dueling DQN, PER DQN, n-step DQN, Rainbow-lite | Main targets |
| **3 — Extended** | A3C, ACER, IMPALA/V-trace, SARSA, Expected SARSA, Q(λ)/SARSA(λ) | After core verified |
| **4 — Stretch** | C51, QR-DQN, MuZero-style, MCTS+learned | Optional/heavy |
| **Excluded** | DDPG, TD3, SAC | Continuous action space, not suitable for discrete 2048 |

---

## 8. Input Representations

The game environment outputs a canonical `(N, 16, 17)` one-hot tensor. A separate representations module transforms this for each algorithm.

### 8.1 Representations Folder Structure

```
representations/
├── base.py            # Common interface
├── onehot.py          # Pass-through (identity)
├── embedding.py       # Learned embeddings
└── cnn.py             # Configurable CNN encoder
```

### 8.2 Interface

```python
class Representation:
    def __init__(self, config):
        """Config holds representation-specific hyperparams."""
        ...

    def forward(self, state: Tensor) -> Tensor:
        """
        Args:
            state: (N, 16, 17) one-hot from env
        Returns:
            Transformed representation for algorithm
        """
        ...

    def output_shape(self) -> tuple:
        """Shape of output for algorithm to know input dim."""
        ...
```

### 8.3 Representation Types

| Type | Description | Output Shape |
|------|-------------|--------------|
| **One-hot** | Pass-through identity | `(N, 16, 17)` |
| **Embedding** | `nn.Embedding(17, embed_dim)` per cell | `(N, 16, embed_dim)` |
| **CNN** | Configurable convolutional encoder | Varies by config |

### 8.4 CNN Hyperparameters

CNN architecture is fully configurable:

| Hyperparam | Description | Examples |
|------------|-------------|----------|
| Kernel shapes | Square or rectangular | `[(4,1), (1,4)]`, `[(3,3)]`, `[(2,2), (4,2), (2,4)]` |
| Features per shape | Feature count per kernel type | `{(4,1): 32, (1,4): 32}` |
| Stride per shape | Stride configuration | Configurable |
| Layers | Depth of network | 1, 2, ... with shape configs per layer |
| Combine method | How to merge multi-shape features | Concat, sum, etc. |

**Kernel shapes and what they capture:**

| Shape | Captures |
|-------|----------|
| 4×1 | Full rows |
| 1×4 | Full columns |
| 4×2 | Half-board horizontal slices |
| 2×4 | Half-board vertical slices |
| 3×3 | Corner/center regions |
| 2×2 | Quadrants |

Multiple kernel shapes can be combined in the same layer (Inception-style).

---

## 9. Experimental Design

### 9.1 Experimental Matrix

Each experimental "try" is a tuple:

```
Try = (Algorithm, Input Representation, Reward Type)
```

| Dimension | Values |
|-----------|--------|
| Algorithm | Per tier list (section 7.3) |
| Representation | one-hot, embedding, CNN (with arch hyperparams) |
| Reward | `merge_reward` (sum of merged tiles), `spawn_reward` (value of spawned tile) |

### 9.2 Hyperparameter Tuning

Each `(Algorithm, Representation, Reward)` combination undergoes systematic hyperparameter tuning.

#### Library: Optuna

| Aspect | Design |
|--------|--------|
| **Library** | Optuna |
| **Storage** | SQLite (local) |
| **Structure** | One study per `(Algorithm, Representation, Reward)` combo |
| **Trials per study** | TBD — will be determined experimentally |
| **Pruning** | Mid-range aggression |
| **Parallelism** | Multiple trials run concurrently on GH200 |

#### Study Structure

Each `(Algorithm, Representation, Reward)` combination gets its own Optuna study. The study tunes:

| Category | Example Hyperparams |
|----------|---------------------|
| Training | learning_rate, batch_size, gamma, n_steps |
| Algorithm-specific | epsilon_decay (DQN), clip_range (PPO), entropy_coef (A2C) |
| Representation | embed_dim, CNN kernel config |
| Architecture | hidden_layer_sizes, activation_fn |

#### Pruning Strategy

Mid-range aggression using MedianPruner:

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,      # Let first 5 trials run fully
    n_warmup_steps=10,       # Don't prune before step 10
    interval_steps=5         # Check every 5 steps
)

study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna.db",
    pruner=pruner
)
```

#### Pruning Integration with Training

Trials report intermediate evaluation scores during training. Poor-performing trials are pruned early:

```python
for epoch in range(n_epochs):
    train_one_epoch()
    eval_score = quick_eval(100_games)
    trial.report(eval_score, epoch)

    if trial.should_prune():
        raise optuna.TrialPruned()
```

---

## 10. Test-First Development Strategy

The game environment will be developed test-first: comprehensive unit tests written BEFORE implementation begins.

### 10.1 Environment Interface (Contract)

```python
class GameEnv:
    def __init__(self, n_games: int, device: torch.device, spawn_fn=None):
        """
        Args:
            n_games: number of parallel games
            device: GPU device
            spawn_fn: optional callable for deterministic tile spawns (testing)
        """

    def reset(self) -> Tensor:
        """Returns initial states (N, 16, 17)"""

    def step(self, actions: Tensor) -> StepResult:
        """
        Args:
            actions: (N,) int 0-3
        Returns:
            StepResult with next_state, done, merge_reward, spawn_reward,
            valid_mask, reset_states
        """
```

### 10.2 Deterministic Spawn Injection (Testing)

For testing, tile spawns can be controlled via an injected function:

```python
def deterministic_spawn(empty_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        empty_mask: (N, 16) boolean mask of empty cells
    Returns:
        positions: (N,) cell index where tile spawns
        values: (N,) tile value (2 or 4)
    """
```

### 10.3 Game Rules (Invariants)

| Rule | Description |
|------|-------------|
| **Merge once** | Each tile merges at most once per move: `[2,2,2,2]` → `[4,4,0,0]`, NOT `[8,0,0,0]` |
| **Merge order** | Merges follow move direction. Left move: left-to-right. `[2,2,2,0]` → `[4,2,0,0]` |
| **Slide after merge** | All tiles slide to fill gaps after merges |
| **Invalid move error** | Env raises exception on invalid move (no board change). Algorithm handles error (penalty, force valid, etc.) |
| **Spawn after valid move** | New tile spawns only after a valid move |
| **Spawn values** | 90% chance of 2, 10% chance of 4 |

### 10.4 Test Categories

| Category | Description |
|----------|-------------|
| **Happy path** | Basic moves, simple merges, standard gameplay |
| **Edge cases** | Full board, single empty cell, almost-terminal |
| **Corner cases** | Max merges in one move, spawning last empty cell |
| **Direction symmetry** | Same scenario rotated, all 4 directions behave correctly |
| **Row/column configs** | `[2,0,2,0]`, `[2,2,0,0]`, `[0,0,2,2]`, etc. |
| **Merge order** | Merges follow move direction (left→right for left move, etc.) |
| **Reward correctness** | merge_reward matches expected sum, spawn_reward is 2 or 4 |
| **Terminal detection** | No valid moves → done=True, valid moves exist → done=False |
| **Rotation invariance** | rotate(board) → move → rotate_back == expected |
| **Reflection invariance** | reflect(board) → move → reflect_back == expected |
| **Invalid move error** | Invalid action raises exception |
| **Episode boundary** | Reset states separate from terminal next_state |
| **GPU-bound (device)** | All tensors on correct device, no CPU tensors |
| **GPU-bound (timing)** | Batched ops complete within time threshold — CPU processing will fail |
| **Precompute correctness** | Line transition, valid move, score delta tables verified |
| **Deterministic spawn** | Injected spawn positions/values work correctly |

### 10.5 Merge Order Test Examples

| Input | Move | Expected | Wrong outputs to reject |
|-------|------|----------|------------------------|
| `[2,2,2,0]` | left | `[4,2,0,0]` | `[2,4,0,0]`, `[4,0,2,0]` |
| `[0,2,2,2]` | left | `[4,2,0,0]` | `[2,4,0,0]` |
| `[2,2,2,2]` | left | `[4,4,0,0]` | `[8,0,0,0]` |
| `[2,2,2,0]` | right | `[0,0,2,4]` | `[0,0,4,2]` |

---

## 11. Milestone 1: Test Suite Creation

**Definition:** Comprehensive test suite for GameEnv — all tests written before implementation begins.

**Deliverables:**

1. Brainstorm all test scenarios (document edge cases, corner cases)
2. Define test fixtures (board states, expected outputs)
3. Write tests using deterministic spawn injection
4. Implement timing harness for GPU verification
5. Tests initially fail (no implementation) or pass against stub/mock

**Success Criteria:**

- All test categories from section 10.4 covered
- Tests are runnable (even if failing against stub)
- GPU timing thresholds calibrated for GH200
- Test fixtures comprehensive enough to catch common implementation bugs
