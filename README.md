# probes-and-varied-rewards-minigrid
CS120 final project

install requirements in requirements.txt

here are a couple templates for cmdline functions that run the scripts in this repo with template flags
**REPLACE THE BRACKETS WITH YOUR PARAMETERS :)

### Training with a specified reward wrapper
```bash
python train_ppo.py \
  --env [door_key, four_rooms]\
  --total-steps [default 200000] \
  --reward-wrapper [subgoal, subgoal_decay, or exploration]  \
  --save-dir [file path] \
  --checkpoint-freq [default 50000]
```

### Resume Training from Checkpoint
```bash
python train_ppo.py \
  --env [door_key, four_rooms] \
  --total-steps [default 200000] \
  --reward-wrapper [subgoal, subgoal_decay, or exploration] \
  --init-model [path to model] \
  --save-dir [file path]
```


### All Available Arguments
- `--env`: Environment name (`door_key`, `four_rooms`)
- `--total-steps`: Total training timesteps (default: 200000)
- `--reward-wrapper`: Reward wrapper type ( `subgoal`, `subgoal_decay`, `exploration`)
- `--reward-wrapper-kw`: Reward wrapper kwargs in `key=value` format
- `--seed`: Random seed (default: 0)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--n-envs`: Number of parallel environments (default: 4)
- `--n-steps`: Steps per environment per update (default: 128)
- `--clip-range`: PPO clip range (default: 0.2)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--gamma`: Discount factor (default: 0.99)
- `--batch-size`: Minibatch size (default: 256)
- `--ent-coef`: Entropy coefficient (default: 0.0)
- `--vf-coef`: Value function coefficient (default: 0.5)
- `--save-dir`: Directory to save models (default: runs)
- `--checkpoint-freq`: Checkpoint frequency in steps (default: 50000)
- `--device`: Device to use (`auto`, `cpu`, `cuda`)
- `--render`: Enable rendering during training
- `--init-model`: Path to checkpoint to resume from

---

## 2. Run a Saved Policy

Visualize rollouts of a trained PPO policy in the environment.

### Basic Policy Evaluation (With Rendering)
```bash
python run_saved_policy.py \
  --model-path [file path] \
  --env [door_key, four_rooms] \
  --episodes [default 3] \
```

### All Available Arguments
- `--model-path`: Path to saved model (required)
- `--env`: Environment name (`door_key`, `four_rooms`)
- `--reward-wrapper`: Optional reward wrapper (`subgoal`, `subgoal_decay`, `exploration`)
- `--reward-wrapper-kw`: Reward wrapper kwargs in `key=value` format
- `--max-steps`: Override max steps per episode
- `--episodes`: Number of episodes to run (default: 1)
- `--seed`: Random seed for episodes
- `--deterministic`: Use deterministic policy actions
- `--no-render`: Disable rendering (headless mode)
- `--device`: Device to load model on (`auto`, `cpu`, `cuda`)

## 3. Generate Evaluation Dataset


### Basic Dataset Generation (Random Actions Only)
```bash
python eval_dataset_gen.py \
  --num-samples [default 2000] \
  --num-episodes [default 200] \
  --output-obs [file path] \
  --output-meta [file path]
```

## 4. Extract Activations from Models

Extract hidden layer activations from trained models on an evaluation dataset.

### Extract Activations from Multiple Models
```bash
python extract_activations.py \
  --eval-obs [npy file with evaluation observations] \
  --models [file path to model] \
  --output-dir [file path]\
  --layer [policy, value] \
  --device [cpu, cuda]
```


## 5. Train Concept Probes on Activations

Train linear probes to detect concepts in model activations.

### Train Probes for Multiple Models
```bash
python train_concept_probes.py \
  --activations-dir [activations filepath] \
  --meta-path [metadata file] \
  --models [paths to models]\
  --output [file path] \
  --test-size [default 0.2]
```
