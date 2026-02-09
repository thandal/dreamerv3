# DreamerV3 Experiment Configurations

This directory contains experiment configurations for testing different design choices in DreamerV3.

## Directory Structure

```
experiments/
├── README.md              # This file
├── baseline.yaml          # Standard DreamerV3 configuration
├── gaussian_latent.yaml   # Use Gaussian instead of categorical latents
└── ablations/             # Ablation studies
    ├── no_symlog.yaml     # Remove symlog transformation
    ├── no_retnorm.yaml    # Remove return normalization
    ├── mse_outputs.yaml   # Use MSE instead of symexp_twohot
    ├── relu_activation.yaml  # Use ReLU instead of SiLU
    └── layer_norm.yaml    # Use LayerNorm instead of RMSNorm
```

## Currently Configurable Design Choices

### 1. Activation Functions (`*.act`)
**Options:** `silu` (default), `relu`, `gelu`, `elu`, `swish`, `mish`

**Example:**
```yaml
agent:
  dyn:
    rssm:
      act: relu  # Use ReLU instead of SiLU
```

**Global replacement:**
```yaml
agent:
  .*\.act: gelu  # Replace all activations
```

### 2. Normalization (`*.norm`)
**Options:** `rms` (default), `layer`, `none`

**Example:**
```yaml
agent:
  .*\.norm: layer  # Use LayerNorm everywhere
```

### 3. Output Distributions (`*.output`)
**Reward/Value Options:** `symexp_twohot` (default), `mse`, `twohot`, `symlog_mse`

**Example:**
```yaml
agent:
  rewhead:
    output: symlog_mse
  value:
    output: symlog_mse
```

### 4. Latent Representation Type (`dyn.rssm.latent`)
**Options:** `onehot` (default), `twohot`

**Example:**
```yaml
agent:
  dyn:
    rssm:
      latent: twohot
```

**Note:** See `gaussian_latent.yaml` for planned Gaussian latent support.

### 5. Return Normalization (`retnorm.impl`)
**Options:** `perc` (percentile, default), `meanstd` (mean/std), `none`

**Example:**
```yaml
agent:
  retnorm:
    impl: none  # Disable return normalization
```

### 6. Optimizer Parameters (`opt.*`)
**Configurable:**
- `lr`: Learning rate (default: 4e-5)
- `agc`: Adaptive gradient clipping (default: 0.3)
- `eps`: Epsilon for numerical stability (default: 1e-20)
- `beta1`: Adam beta1 (default: 0.9)
- `beta2`: Adam beta2 (default: 0.999)
- `wd`: Weight decay (default: 0.0)

**Example:**
```yaml
agent:
  opt:
    lr: 1e-4
    agc: 0.5
    beta1: 0.5
```

### 7. Loss Scales (`loss_scales.*`)
**Configurable:** All β coefficients for different loss terms

**Example:**
```yaml
agent:
  loss_scales:
    rec: 1.0   # Reconstruction loss
    rew: 1.0   # Reward prediction loss
    con: 1.0   # Continuation prediction loss
    dyn: 1.0   # Dynamics KL loss
    rep: 0.1   # Representation KL loss
    policy: 1.0
    value: 1.0
    repval: 0.3
```

### 8. Policy Distribution Types
**Discrete Actions:** `categorical` (default), `onehot`
**Continuous Actions:** `bounded_normal` (default), `normal`, `tanh_normal`

**Example:**
```yaml
agent:
  policy_dist_disc: onehot
  policy_dist_cont: tanh_normal
```

### 9. TD(λ) Parameter (`imag_loss.lam`, `repl_loss.lam`)
**Range:** 0.0 to 1.0 (default: 0.95)

**Example:**
```yaml
agent:
  imag_loss:
    lam: 0.99  # More bootstrapping
  repl_loss:
    lam: 0.99
```

### 10. Model Sizes
**Presets:** `size1m`, `size12m`, `size25m`, `size50m`, `size100m`, `size200m`, `size400m`

**Example:**
```bash
python main.py --configs size12m dmc_proprio --task dmc_walker_walk
```

### 11. Symlog Transformation (`enc.simple.symlog`)
**Options:** `True` (default), `False`

**Example:**
```yaml
agent:
  enc:
    simple:
      symlog: False
```

### 12. RSSM Architecture
**Configurable:**
- `deter`: Deterministic state size
- `hidden`: Hidden layer size
- `stoch`: Number of stochastic variables
- `classes`: Classes per variable (for categorical)
- `blocks`: Number of GRU blocks

**Example:**
```yaml
agent:
  dyn:
    rssm:
      deter: 4096
      hidden: 512
      stoch: 16
      classes: 32
      blocks: 4
```

## Usage Examples

### Run baseline training:
```bash
python main.py --configs debug dmc_proprio --task dmc_walker_walk
```

### Run with Gaussian latents (future):
```bash
python main.py --configs debug dmc_proprio gaussian_latent --task dmc_walker_walk
```

### Run ablation study:
```bash
python main.py --configs debug ablations/no_symlog --task dmc_walker_walk
```

### Combine multiple configs:
```bash
python main.py --configs debug size12m ablations/relu_activation --task dmc_walker_walk
```

## Planned Extensions (Phase 3.2+)

The following design choices will be made configurable in upcoming phases:

### Phase 3.2: Latent Types
- [ ] Gaussian latents (`latent: gaussian`)
- [ ] Mixture of Gaussians (`latent: mixture`)

### Phase 3.3: Sequence Models
- [ ] Standard GRU (`seq_model: gru`)
- [ ] LSTM (`seq_model: lstm`)
- [ ] Transformer (`seq_model: transformer`)

### Phase 3.4: Encoder/Decoder Architectures
- [ ] ResNet encoder/decoder
- [ ] Vision Transformer (ViT) encoder

### Phase 3.5: Loss Functions
- [ ] Contrastive world model loss
- [ ] N-step return loss
- [ ] Monte Carlo return loss

### Phase 3.6: Return Computation
- [ ] Generalized Advantage Estimation (GAE)
- [ ] N-step returns
- [ ] Monte Carlo returns

## Contributing New Experiments

To add a new experiment configuration:

1. Create a new YAML file in this directory (or `ablations/`)
2. Add a descriptive comment block explaining what it tests
3. Override only the parameters that differ from defaults
4. Update this README with the new configuration

Example template:
```yaml
# Experiment: Your experiment name
# Description: What this tests and why it's interesting
# Reference: Paper citation if applicable

your_experiment_name:
  agent:
    your_override:
      param: value
```
