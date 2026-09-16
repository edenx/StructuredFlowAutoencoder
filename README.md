# SFA: Stochastic Flow Analysis

Flow matching framework for generative modeling with mixture latent variables across multiple data modalities.

## Project Structure

```
SFA/
├── src_sfa/                # Main source code
│   ├── main.py             # Entry point (training & evaluation)
│   ├── sfa_llk.py          # Unified vector field (LLK) — supports mlp, cnn, cnn_film, resnet, unet
│   ├── sfa.py              # Continuous priors (GaussianPrior, CNF, FlowMatchingLoss)
│   ├── sfa_discrete.py     # Mixture components (CatNF_fixed, GaussianMixtureComponent, FlowMatchingLossMixture)
│   ├── sfa_lds.py          # Sequence models (LatentDynamicalSystem, fullCNF, FlowMatchingLossSeq)
│   ├── runner_continuous.py    # Runners: PinWheel, MNIST, HVG (continuous latent)
│   ├── runner_mixture.py       # Runners: PinWheel, MNIST (mixture latent)
│   ├── runner_pendulum.py      # Runner: Pendulum (sequential latent)
│   ├── callbacks.py        # Lightning callbacks for plotting
│   ├── Scheduler.py        # Learning rate schedulers (warmup)
│   └── utils.py            # Shared utilities (PCA, t-SNE, metrics)
├── configs/                # YAML config files per experiment
├── models/                 # Neural network architectures (MLP, UNet)
└── dataloader/             # Dataset loaders (pinwheel, MNIST, HVG, pendulum, etc.)
```

## Runners

| Runner | Config | Data | Latent |
|--------|--------|------|--------|
| `pinwheel` | `pinwheel.yml` | 2D synthetic pinwheel | Continuous z |
| `mnist` | `mnist.yml` | MNIST images | Continuous z |
| `hvg` | `hvg.yml` | Highly variable genes | Continuous z |
| `pinwheel_mixture` | `pinwheel_discrete.yml` | 2D synthetic pinwheel | Mixture (pi, z) |
| `mnist_mixture` | `mnist.yml` | MNIST images | Mixture (pi, z) |
| `pendulum` | `pendulum.yml` | Pendulum image sequences | Sequential z(t) |

## Training

```bash
cd src_sfa

# MNIST with mixture latent
python main.py --config mnist.yml --runner mnist_mixture --ni

# MNIST with continuous latent
python main.py --config mnist.yml --runner mnist --ni

# Pinwheel with mixture latent
python main.py --config pinwheel_discrete.yml --runner pinwheel_mixture --ni

# Pinwheel with continuous latent
python main.py --config pinwheel.yml --runner pinwheel --ni

# Pendulum sequences
python main.py --config pendulum.yml --runner pendulum --ni
```

`--ni` suppresses interactive prompts (auto-overwrites existing logs). Logs and checkpoints are saved to `exp/logs/<runner_name>/`.

### Resume training

```bash
python main.py --config mnist.yml --runner mnist_mixture --resume_training --ni
```

### Custom log folder

```bash
python main.py --config mnist.yml --runner mnist_mixture --doc my_experiment --ni
```

## Evaluation

```bash
# Test / evaluation
python main.py --config mnist.yml --runner mnist_mixture --test --ni

# Generate figures
python main.py --config mnist.yml --runner mnist_mixture --figure --ni
```

## Key Config Options

In `configs/*.yml`:

- `model.type`: LLK architecture — `"cnn"` or `"mlp"`
- `flow.z_dim`: Latent dimension
- `flow.pi_dim`: Number of mixture components (mixture runners)
- `flow.beta`: Softmax temperature for mixture assignment
- `training.n_epochs`: Number of training epochs
- `training.alpha`: Regularization weight
- `optim.lr`: Learning rate

## Setup

```bash
conda env create -f environment.yml
conda activate my_env
```
