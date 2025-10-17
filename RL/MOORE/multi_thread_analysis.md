# Multi-Processing Analysis: MOORE RL Codebase

## Overview

This codebase uses **Python multiprocessing** (not multi-threading) to parallelize robotic simulation environments. Each environment runs in its own separate process, allowing parallel execution of multiple Metaworld/MiniGrid tasks simultaneously.

---

## 1. CORE MULTI-PROCESSING: SubprocVecEnv

### Block 1: Process Creation
**File**: `moore/environments/subproc_vec_env.py:83-112`

```python
def __init__(self, env_fns, start_method=None):
    n_envs = len(env_fns)  # Number of processes = number of tasks

    # Create bidirectional communication pipes (one per environment)
    self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])

    # Launch one process per environment
    self.processes = []
    for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
        args = (work_remote, remote, CloudpickleWrapper(env_fn))
        process = ctx.Process(target=_worker, args=args, daemon=True)
        process.start()  # ← Process starts here
        self.processes.append(process)
```

**How many processes?** `n_envs` = number of tasks
- **MT5**: 5 processes
- **MT10**: 10 processes
- **MT50**: 50 processes

Each process runs the `_worker` function independently.

---

### Block 2: Worker Function
**File**: `moore/environments/subproc_vec_env.py:16-56`

This runs in **each separate process**:

```python
def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()  # Close parent's end of pipe
    env = env_fn_wrapper.var()  # Create environment instance

    while True:  # Infinite loop waiting for commands
        cmd, data = remote.recv()  # Wait for command from main process

        if cmd == 'step':
            observation, reward, absorbing, info = env.step(data)
            remote.send((observation, reward, absorbing, info))
        elif cmd == 'reset':
            observation = env.reset(data)
            remote.send(observation)
        # ... other commands (seed, render, close, stop, etc.)
```

**Each worker:**
- Runs in its own CPU process
- Has its own copy of the environment
- Waits for commands via pipe
- Sends results back via pipe

---

### Block 3: Async Step Pattern
**File**: `moore/environments/subproc_vec_env.py:122-132`

This is how parallel execution works:

```python
def step_async(self, actions):
    # Send actions to ALL processes (non-blocking)
    for remote, action in zip(self.remotes, actions):
        remote.send(('step', action))  # ← Sends to all workers
    self.waiting = True

def step_wait(self):
    # Collect results from ALL processes
    results = [remote.recv() for remote in self.remotes]  # ← Waits for all
    obs, rews, dones, infos = zip(*results)
    return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
```

**This allows parallel stepping:**
1. `step_async()` sends actions to all 50 processes simultaneously
2. All 50 environments step in parallel (each on different CPU)
3. `step_wait()` collects all 50 results

---

### Block 4: Start Method Configuration
**File**: `moore/environments/subproc_vec_env.py:92-102`

```python
start_method = os.environ.get("DEFAULT_START_METHOD")  # Can override via env var

if start_method is None:
    forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
    start_method = 'forkserver' if forkserver_available else 'spawn'

ctx = multiprocessing.get_context(start_method)
```

**Start methods:**
- **`forkserver`** (preferred): Thread-safe, works with PyTorch/TensorFlow
- **`spawn`** (fallback): Slower but universal
- **NOT `fork`**: Not thread-safe, avoided

**To override**: Set environment variable `DEFAULT_START_METHOD` before running:
```bash
export DEFAULT_START_METHOD=spawn
```

---

## 2. HOW IT'S USED: Metaworld Training

### Block 5: Environment Setup
**File**: `run_metaworld_sac_mt.py:48-58`

```python
benchmark = getattr(metaworld, exp_type)()  # e.g., MT50()

# Create 50 parallel environments (one per task)
mdp = SubprocVecEnv(
    [make_env(env_name=env_name,
              env_cls=env_cls,
              train_tasks=benchmark.train_tasks,
              horizon=horizon,
              gamma=gamma,
              normalize_reward=args.normalize_reward,
              sample_task_per_episode=args.sample_task_per_episode)
     for env_name, env_cls in benchmark.train_classes.items()])

n_contexts = mdp.num_envs  # 50 for MT50
```

**For MT50:**
- Creates list of 50 environment functions
- `SubprocVecEnv` spawns **50 separate processes**
- Each process runs one Metaworld task (e.g., assembly-v2, basketball-v2, etc.)

---

## 3. PARALLEL STEPPING: VecCore

### Block 6: Vectorized State Management
**File**: `moore/core/vec_core.py:7-15`

```python
class VecCore(object):
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None):
        self.agent = agent
        self.mdp = mdp
        self._n_mdp = self.mdp.num_envs  # Number of parallel environments

        # State array for ALL environments
        self._state = np.zeros((self._n_mdp, self.mdp.info.observation_space.shape[0]))
```

Maintains state for all 50 environments simultaneously in vectorized arrays.

---

### Block 7: Parallel Step Execution
**File**: `moore/core/vec_core.py:243-261`

```python
def _step(self, render):
    idx = np.arange(self._n_mdp)  # [0, 1, 2, ..., 49] for MT50

    # Get actions for all 50 environments at once
    action = self.agent.draw_action([idx, np.stack(self._state, axis=0)])

    # Step all 50 environments in parallel
    next_state, reward, absorbing, step_info = self.mdp.step(action)

    # Update all 50 states
    self._state = self._preprocess(next_state.copy())

    # Create 50 transitions
    sample = [([i, state[i]], action[i], reward[i], [i, next_state[i]],
               absorbing[i], last[i]) for i in range(self._n_mdp)]

    return sample, last, step_info
```

This calls `mdp.step(action)` which internally:
1. Calls `step_async()` → sends 50 actions to 50 processes
2. All 50 processes execute `env.step()` in parallel
3. Calls `step_wait()` → waits for 50 results
4. Returns vectorized results

---

## 4. CPU/THREAD COUNT

### How Many CPUs Are Used?

**For MT50 Metaworld:**
- **50 CPU processes** (one per task/environment)
- Plus **1 main process** (for coordinating and training neural networks)
- **Total: 51 processes actively using CPUs**

**Verification from code:**
```python
# run_metaworld_sac_mt.py:58
n_contexts = mdp.num_envs  # This determines process count
```

### Process Hierarchy

```
Main Process (Python)
│
├── Process 1: assembly-v2 (CPU/Core usage)
├── Process 2: basketball-v2 (CPU/Core usage)
├── Process 3: bin-picking-v2 (CPU/Core usage)
├── Process 4: box-close-v2 (CPU/Core usage)
...
└── Process 50: window-open-v2 (CPU/Core usage)

Total: 51 processes (1 main + 50 workers)
```

---

## 5. MACHINE CONFIGURATION

**Current machine: 30 CPU cores** (verified via `nproc`)

### Implications for Different Setups:

#### MT50 Setup (50 environments)
```
50 environment processes + 1 main process = 51 total processes
30 physical CPU cores
→ ~1.7 processes per core (OVERSUBSCRIBED)
→ Context switching overhead
```

#### MT10 Setup (10 environments)
```
10 environment processes + 1 main process = 11 total processes
30 physical CPU cores
→ ~0.37 processes per core (OPTIMAL)
→ Good CPU utilization without oversubscription
```

#### MT5 Setup (5 environments)
```
5 environment processes + 1 main process = 6 total processes
30 physical CPU cores
→ ~0.2 processes per core (UNDERUTILIZED)
→ Could run more environments efficiently
```

---

## 6. PERFORMANCE CONSIDERATIONS

### From Code Documentation
**File**: `moore/environments/subproc_vec_env.py:63-64`

```python
# For performance reasons, if your environment is not IO bound,
# the number of environments should not exceed the number of logical cores on your CPU.
```

### Current Status
- **MT50 on 30 cores**: Oversubscribed by ~1.67x
- **Recommendation**: For optimal performance on this machine, use MT10 or MT30
- **GPU Usage**: Neural network training uses GPU (`--use_cuda`), environments use CPU

### Trade-offs
- **More processes than cores**: Context switching overhead, but more task diversity
- **Fewer processes than cores**: Better CPU efficiency, but less task diversity
- **For MT50**: Accept some overhead for training on all 50 tasks simultaneously

---

## 7. ADDITIONAL PARALLELIZATION: Joblib

### Experiment-Level Parallelization
**File**: `run_minigrid_ppo_mt.py:271-276`

```python
from joblib import delayed, Parallel

# Run multiple experiments (different seeds) in parallel
if args.seed is not None:
    out = Parallel(n_jobs=-1)(delayed(run_experiment)(args, save_dir, i, s)
                          for i, s in zip(range(args.n_exp), args.seed))
```

**Configuration:**
- `n_jobs=-1`: Uses all available CPU cores (30 on this machine)
- Runs multiple experiments (different random seeds) in parallel
- Each experiment is independent and runs on separate CPU cores

**Note**: This is typically used for MiniGrid experiments, not Metaworld (which uses single experiment runs)

---

## 8. KEY ARCHITECTURAL DECISIONS

### Communication Method
- **IPC**: `multiprocessing.Pipe()` for bidirectional communication
- **Serialization**: CloudPickle for handling complex environment functions
- **Pattern**: Producer-consumer with command-response protocol

### Thread Safety
- **Start Method**: Prefers `forkserver` over `fork`
- **Reason**: Compatible with PyTorch/TensorFlow (non-thread-safe libraries)
- **Fallback**: Uses `spawn` if `forkserver` unavailable

### Synchronization
- **Async Pattern**: Non-blocking send, blocking receive
- **Wait for All**: `step_wait()` blocks until all processes respond
- **No Partial Results**: Always processes all environments together

---

## 9. CODE FLOW SUMMARY

### Training Loop Flow
```
VecCore.learn()
  → VecCore._run_impl()
    → VecCore._step()
      → agent.draw_action()  [get actions for all 50 envs]
      → mdp.step(actions)
        → SubprocVecEnv.step_async()  [send to 50 processes]
        → [50 workers execute env.step() in parallel]
        → SubprocVecEnv.step_wait()  [receive from 50 processes]
      → Returns vectorized [states, rewards, dones, infos]
    → agent.fit(dataset)  [train neural networks on collected data]
```

### Evaluation Loop Flow
```
VecCore.evaluate()
  → VecCore._run_eval_impl()
    → VecCore._eval_step()  [evaluates ONE task at a time]
      → mdp.env_method("step", action, indices=i)
      → Returns single transition for task i
```

**Note**: Training uses all processes in parallel, evaluation uses them sequentially (one at a time).

---

## 10. CONFIGURATION FILES

### Example: MT50 Shell Script
**File**: `run/metaworld/run_metaworld_mt50_mhsac_mt_moore_GRIN.sh`

```bash
python run_metaworld_sac_mt.py \
    --exp_type MT50 \              # 50 parallel environments
    --batch_size 128 \             # Neural network batch size
    --n_epochs 20 \                # Training epochs
    --n_steps 100000 \             # Steps per epoch
    --use_cuda \                   # GPU for neural networks
    --n_episodes_test 10 \         # Evaluation episodes per task
    ...
```

**Key Parameters:**
- `--exp_type MT50`: Determines number of parallel processes (50)
- `--use_cuda`: GPU acceleration for neural network training
- `--batch_size 128`: Training batch size (independent of process count)

---

## 11. MONITORING AND DEBUGGING

### Check CPU Usage During Training
```bash
# Monitor CPU usage in real-time
htop

# Check process count
ps aux | grep python | wc -l

# Monitor specific processes
watch -n 1 'ps aux | grep run_metaworld_sac_mt'
```

### Environment Variable Override
```bash
# Force spawn start method (for debugging)
export DEFAULT_START_METHOD=spawn
python run_metaworld_sac_mt.py --exp_type MT50 ...

# Verify start method
python -c "import multiprocessing; print(multiprocessing.get_all_start_methods())"
```

---

## 12. OPTIMIZATION RECOMMENDATIONS

### For This 30-Core Machine

#### Option 1: Reduce to MT30
- Modify benchmark to use 30 tasks instead of 50
- One process per core (optimal)
- Trade-off: Less task diversity

#### Option 2: Keep MT50
- Accept 1.7x oversubscription
- Context switching overhead ~15-30%
- Benefit: Full task coverage

#### Option 3: Batch Processing
- Train on 30 tasks at a time
- Rotate through all 50 tasks
- Best CPU efficiency, longer training time

### Code Modification Example
To limit environments programmatically:

```python
# In run_metaworld_sac_mt.py, after line 46
benchmark = getattr(metaworld, exp_type)()

# Limit to first 30 tasks
max_envs = 30
task_items = list(benchmark.train_classes.items())[:max_envs]

mdp = SubprocVecEnv(
    [make_env(env_name=env_name, env_cls=env_cls, ...)
     for env_name, env_cls in task_items])
```

---

## 13. SUMMARY TABLE

| Mechanism | Files | Type | Configuration | Scale |
|-----------|-------|------|---------------|-------|
| SubprocVecEnv | `subproc_vec_env.py` | Multiprocessing | `n_envs = num_tasks` | MT5-MT50 (5-50 processes) |
| VecEnv Pipes | `base_vec_env.py` | IPC | Async step_async/wait | Per environment |
| VecCore | `vec_core.py` | Parallel stepping | `num_envs` from SubprocVecEnv | Vectorized arrays |
| Joblib Parallel | `run_*_mt.py` | Multiprocessing | `n_jobs=-1` | `n_exp` experiments |
| Context Replay | `mtsac.py` | Per-task buffers | `n_contexts` replay buffers | Same as `num_envs` |

---

## 14. FREQUENTLY ASKED QUESTIONS

### Q: Is this multi-threading or multi-processing?
**A**: Multi-processing. Each environment runs in a separate Python process, not thread.

### Q: How many CPUs should I have?
**A**: Ideally, at least as many as the number of environments (5 for MT5, 10 for MT10, 50 for MT50).

### Q: Does it use GPU?
**A**: Yes, with `--use_cuda` flag. GPU is used for neural network training, CPUs for environment simulation.

### Q: Can I change the number of processes?
**A**: Yes, by changing `--exp_type` (MT5, MT10, MT50) or modifying the benchmark task list.

### Q: What if I have fewer cores than environments?
**A**: It will still work, but with context switching overhead. Performance degradation is typically 15-30%.

### Q: Can I monitor individual processes?
**A**: Yes, use `htop` or `ps aux | grep python` to see all worker processes.

---

## 15. REFERENCES

- **Stable Baselines3**: Original implementation of SubprocVecEnv
  - https://github.com/DLR-RM/stable-baselines3
- **Python Multiprocessing**: Official documentation
  - https://docs.python.org/3/library/multiprocessing.html
- **CloudPickle**: Serialization library
  - https://github.com/cloudpipe/cloudpickle

---

## Appendix: Process Communication Protocol

### Command Protocol
Each worker process accepts these commands:

| Command | Data | Response | Description |
|---------|------|----------|-------------|
| `'step'` | action | (obs, reward, absorbing, info) | Step environment |
| `'reset'` | initial_state | observation | Reset environment |
| `'seed'` | seed | seed_result | Set random seed |
| `'render'` | mode | rendered_image | Render environment |
| `'close'` | None | None | Close and terminate |
| `'stop'` | None | None | Stop environment |
| `'get_spaces'` | None | (obs_space, action_space) | Get spaces |
| `'get_mdp_info'` | None | mdp_info | Get MDP info |
| `'env_method'` | (method, args, kwargs) | method_result | Call env method |
| `'get_attr'` | attr_name | attr_value | Get attribute |
| `'set_attr'` | (attr_name, value) | None | Set attribute |

All communication happens via `multiprocessing.Pipe()` objects.
