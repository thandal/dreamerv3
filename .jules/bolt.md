## 2025-03-05 - Optimize JAX Compilation via jax.lax.scan
**Learning:** Python `for` loops in JAX functions (especially sequence computations like backward passes for returns) cause extreme compilation times due to HLO graph unrolling. In `dreamerv3/returns.py`, the compilation took over 4 minutes for a sequence length of 500.
**Action:** Always replace Python loops over the time dimension with JAX-native operations like `jax.lax.scan` (using `reverse=True` for backward passes). This keeps the node graph size O(1), drastically improving both compilation speed and memory footprint.
