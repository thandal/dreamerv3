## 2024-03-09 - Use jax.lax.scan instead of Python loops for sequence computation in JAX
**Learning:** Python loops over the time dimension (e.g., in backward passes for returns or GAE) result in O(T) node graph size during `jax.jit` compilation due to HLO graph unrolling. This leads to drastically slower compilation speeds and much higher memory footprint.
**Action:** Replace Python loops with JAX-native sequence operations like `jax.lax.scan` (using `reverse=True` for backward passes) to ensure the node graph size remains O(1) during compilation.
