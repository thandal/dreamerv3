## 2024-05-18 - Replacing Python sequence loops with jax.lax.scan
**Learning:** In JAX-based applications, performing sequence computations (like iterating over time horizons for returns or advantages) using Python `for` loops results in HLO graph unrolling. This leads to massive compilation times and large memory overhead scaling at O(T).
**Action:** Always replace Python sequence loops in JAX functions with `jax.lax.scan` (or `vmap` when independent). `jax.lax.scan` allows JAX to compile the loop operation in O(1) time and memory relative to the sequence length.
