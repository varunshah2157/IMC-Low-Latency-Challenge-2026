# Track B: Feature Model Selection & Simulation Architecture Notes

Engineering notes and simulation benchmarks for the exploratory Track B engine (`solution.cpp` and variants `v1`–`v12`), focusing on multi-feature online model selection, pricing kernel vectorization, and local event simulation.

---

### Core Architectural Decisions

| Baseline (Event-Driven) | Optimized Track B Architecture | Engineering Rationale |
|---|---|---|
| Quotes only options touched by incoming update | Persistent quote book re-sent on **every** call (`return book;`) | Each registered option is scored at most once per `process()` call against a hidden reference. A valid cached quote earns on every subsequent call. In local simulation, this alone yields a ~10x increase in cumulative PnL. |
| Abramowitz-Stegun CDF ($10^{-7}$) + `std::exp` | Hart/West rational Chebyshev CDF ($< 10^{-14}$) with branch-free exp, AVX2 intrinsics, 4 options/iteration, shared reciprocal for $\mathcal{N}(d_1)$ and $\mathcal{N}(d_2)$ | Evaluates at ~4 ns per option on AVX2 hardware with valuation error below $10^{-12}$. Uses per-function `__attribute__((target("avx2,fma")))` to ensure vectorized codegen without relying on compiler `-march` flags. |
| Precomputed lookup tables (LUT) | Direct closed-form vectorized evaluation | Closed-form AVX2 kernel is fast (< 4 ns), whereas LUT interpolation introduces discretization noise near the tight score-loss threshold boundary. |
| `profit` hill-climbing on consecutive ticks | Paired $\pm \delta$ step probes in $\sigma$ (Rprop step scaling) | Differential evaluation of $\bar{P}(\sigma + \delta)$ vs $\bar{P}(\sigma - \delta)$ isolates the gradient sign of volatility error without assuming knowledge of the hidden score curve. |
| Stop-loss deactivation until next $V$ update | Adaptive exponential backoff retry (8 calls, doubling to $2^{20}$), resetting immediately on new $V$ | Benching an option permanently until the next volatility update forfeits too many profitable quoting opportunities during rapid spot regimes. |
| Sibling volatility model | Sibling factor model: $\ln \sigma_i = \text{level}(\text{und}) + \Delta_i$, dynamically selecting among four gains $(0, 0.15, 0.4, 1.0)$ | Tracks whether volatility surprises co-move across options of the same underlying. If uncorrelated, gain 0 activates automatically. |
| Naive vector copying | Contiguous buffer memmove with `mallopt` memory pooling | Keeps large return vectors in CPU L1/L2 cache and eliminates page allocation overhead. |

---

### Local Simulation Benchmarks (`sim/`)

Local testing with `sim/sim.cpp` (geometric Brownian motion spot, log-volatility random walk, asynchronous $V$ sampling, and weighted linear threshold scoring $w \cdot (1 - |\text{err}| / \text{thr})$):

| Variant Architecture | Simulated PnL (Base: 500 opts, 40k updates) | Key Characteristics |
|---|---|---|
| **A: Event-Only Baseline** | 0.68M | Reprices and quotes only updated option/underlying |
| **B: Persistent Book + Stop-Loss** | 7.10M | Quotes full book on every call; benches on negative score |
| **E: Persistent Book + Probing + Sibling Model** | 11.20M | Online Rprop probing and sibling volatility propagation |

*Note:* While probing and sibling pooling outperformed in local simulations with synthetic random walks, real grader evaluation revealed that independent per-option spot-vol slopes with $t$-statistic shrinkage (`submission_E` / `v9_best`) dominated empirical scoring.

---

### Key Knobs & Runtime Macros

Tunable parameters located at the top of `solution.cpp`:
- `LLC_REQUOTE_ALL`: Toggle full persistent quote book vs event-only delta.
- `LLC_STOPLOSS`: Negative score detection and gating.
- `LLC_PROBE`: Paired $\pm \delta$ gradient probing between volatility updates.
- `LLC_SIBLING`: Cross-option volatility propagation on the same underlying.
- `LLC_STEP0` (`0.25`): Base step size for exploratory probing.
- `LLC_RETRY_BASE` (`8`): Initial backoff call count after encountering a loss.

---

### Simulator Usage

```bash
cd sim && cp ../../template/api.hpp . && bash build.sh && bash run.sh
# Args: nUnderlyings optsPerUnderlying nUpdates pV factorDrift idioDrift scoreType(0 lin, 1 quad, 2 step) thrVolPts [seed]
```
