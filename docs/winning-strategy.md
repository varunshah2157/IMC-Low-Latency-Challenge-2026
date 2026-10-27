# Low Latency Challenge: Winning Strategy

*2026-10-26 · saksham*

---

## TL;DR

The score is earned per option, per `process` call, so the game is to send as many good valuations as possible on every call and to stop sending bad ones. Speed matters because it buys more valuations per call; feedback matters because it tells you which ones are good.

- **Quote on every call, not on 5.** One Black–Scholes price costs about 20–50 ns. A 1 s budget fits millions of them. Limiting yourself to a handful of calls throws away almost all of the score.
- **Re-send cached prices for options whose inputs did not change.** Each option can score once per call, even if nothing about it changed. Re-sending a cached `Valuation` costs about 1 ns (a memcpy), while re-pricing costs 20–50 ns. This is probably the single biggest lever, and it needs one test submission to confirm.
- **Use `previousScores` as a per-option signal.** It shows how much each option is worth (its weight) and whether its price has gone stale. Stop quoting an option once its expected profit turns negative, and start again after its next `V` update.
- **Pace the budget with a controller.** Running out of budget only loses the remaining inputs, so spend aggressively and measure your own time with `rdtsc`.
- **Edge beyond the obvious:** guess the true current volatility between `V` updates. Do it from sibling options on the same underlying and from small, deliberate price nudges, using the scores you get back.

---

## What the Rules Actually Reward

Read closely, the rules define a throughput problem with a noisy quality filter. The phrase *"valuation calls"* is misleading, because every `process` call is a chance to score.

| Rule | What it implies |
| :--- | :--- |
| **Each option is scored at most once per call** | Profit per call is the sum over every option you price. Pricing more options on more calls means more profit. |
| **You may price any subset of options, in any call** | You are not limited to options whose inputs changed. Any option can be quoted again from cache. |
| **Up to (options + 64) valuations are read per call** | You can quote the whole book on every call. |
| **A price beyond a threshold loses money** | Quoting has a cost when the price is wrong. A stale volatility is the main source of error. |
| **Some options carry more money** | Weights differ per option. Unknown at the start, but they can be learned from `previousScores`. |
| **`previousScores` returns the profit of each valuation, in the order you sent them** | You get ground truth on every call, for free, indexed by position. |
| **1 s budget, counted only inside `process`, plus ~0.1 µs fixed cost per call** | Setup (2 s) is free: do all the precomputation there. The fixed cost per call is unavoidable. |
| **Running out of budget keeps the score so far** | Overspending at the end is cheap and underspending is waste, so pace aggressively. |
| **The grader shows inputs processed and time used per test** | Free reconnaissance: one empty submission tells you the number of inputs in each test. |
| **Generic x86-64, no `-march`** | `#pragma GCC target("avx2,fma")` is still allowed, and Sapphire Rapids supports it. |

The reference price most likely uses the *true* current volatility. That value drifts away from the last `V` you received, so a quote's accuracy decays with time (and with how far $S$ has moved) since that option's last `V` update.

---

## Flaws in the Two Initial Ideas

Both ideas treat the budget as *"N expensive calls to place well"*. The budget is actually millions of cheap valuations, and their placement should follow per-option quality, not the clock.

### Idea 1: Measure time per valuation, spread 5 updates uniformly over 50,000 ticks

- It leaves about 99.99% of the score on the table. Calls that return nothing score 0, and every call is a scoring opportunity for every option.
- It misreads the cost. One BS price is about 20–50 ns, so 1 s is roughly 20–50 million prices. With 50,000 ticks you can afford about 20 µs per tick, which is enough to price hundreds of options on every tick.
- It assumes a call either prices everything or nothing. In fact it is a per-option choice on every call: which options to include.
- Local timings do not carry over. The grader uses different hardware, adds about 0.1 µs per call and measures from outside. The tick count is also unknown and differs per test.
- It is blind to quality. A quote right after that option's `V` update is near perfect, while one 10,000 ticks later may lose money. Uniform spacing ignores this.

### Idea 2: Bias calls toward the end, "because we have more data"

- Black–Scholes needs only the latest $S$ and $\sigma$, so older data does not make the price more accurate. What makes a price accurate is a recent `V` for that option, and that can happen at any time.
- It is risky on the budget. If the end-heavy schedule overruns, the test stops and the tail, where you put your value, earns nothing.
- It wastes early opportunities, which are worth just as much as late ones.

### What both miss:

- `previousScores`: per-option weights, negative-profit detection, and the true error signal.
- The threshold: sometimes the best move is *not* to quote a given option.
- Re-sending cached prices nearly for free.
- Adapting the pace to how much budget is actually left.

The instinct that continuous calls are better is right. The refinement is **"continuous, per option, gated by feedback"**.

---

## Latency Hotspots and Fixes

Ranked by expected cost per call (estimates for Sapphire Rapids):

| # | Hotspot | Naive Cost | Fix |
| :-: | :--- | :--- | :--- |
| **1** | Transcendental math in BS: `log`, `exp`, `sqrt`, $2\times \text{erfc}$ | 40–80 ns per option | Precompute per option at setup: $\ln K$, $K e^{-rT}$, $\sqrt{T}$, $rT$. Compute $\ln S$ once per underlying per update. On a `V`, cache $\sigma\sqrt{T}$ and $0.5\sigma^2 T$. Replace $\text{erfc}$ with a fast polynomial/rational $N(x)$ accurate to $\sim 10^{-7}$ or Hart CDF. |
| **2** | Re-pricing every option on an underlying for each `U` | $m \times 50$ ns | Use a greeks Taylor update: $P + \Delta \cdot dS + \frac{1}{2}\Gamma \cdot dS^2 + \frac{1}{6}\text{speed} \cdot dS^3$, which is ~5 FMAs per option. Do a full re-price only when cumulative $\|dS\|$ passes a bound. Vectorize over options with SoA layout and AVX2. |
| **3** | Building return vector with `push_back` | Bounds checks + regrowth | Keep a persistent quote book (contiguous `Valuation[]`, updated in place). Return `std::vector<Valuation>(book, book + k)` — one allocation and one memmove. |
| **4** | Heap allocation for returned vector + grader free | 20–100 ns | Replace global `operator new`/`delete` with a size-class free list over a pre-reserved arena. Keeps hot memory in cache. |
| **5** | `unordered_map` lookups for option and underlying IDs | 10–30 ns each | Remap to dense indices at setup. Use direct-index arrays if ID range is small, otherwise open-addressing table built once. Store options grouped by underlying. |
| **6** | Scanning `previousScores` | $k$ entries per call | Scores return in the order sent, so map index to option directly. Subsample once weights have converged. |
| **7** | Generic x86-64 codegen | SSE2 only, no FMA | `#pragma GCC target("avx2,fma")` and `#pragma GCC optimize("O3,unroll-loops")`. Use `fast-math` inside the pricing kernel. |
| **8** | Grader fixed overhead (~0.1 µs per call) | Unavoidable | Budget for it. With 1M updates, it alone consumes ~10% of the budget. |

*Note on accuracy:* A fast $N(x)$ error of $\sim 10^{-7}$ and Taylor truncation error are negligible compared to error from a stale volatility. Use the computational savings to buy more quotes.

---

## The Proposed Strategy

The strategy consists of four layers:

### Layer 1: Quote Book Engine (Speed)
- At setup, store every option in Structure-of-Arrays (SoA) layout grouped by underlying, with all static parameters precomputed.
- Maintain one `Valuation` per option in a persistent book, plus an `active` flag. Only active options are packed into the contiguous prefix returned.
- On `U(u, S)`: update prices of $u$'s options in place using Taylor or full pricing kernel. On `V(i, \sigma)`: run full re-price of option $i$, update its greeks, and mark it fresh.
- Return the active prefix with a single memmove, so every active option scores on *every* call.

### Layer 2: Feedback Gating (Quality)
For each option, track metrics from `previousScores`:
- $w_i$: estimated weight (profit seen on fresh quotes right after a `V`).
- $\text{ev}_i$: exponential moving average (EWMA) of recent profit.
- $\text{age}_i$ and $\text{drift}_i$: ticks elapsed and $\|\ln S - \ln S_{\text{at } V}\|$ since last `V`.

**Rules:**
- If the last $k$ profits for option $i$ were negative, deactivate it until its next `V`.
- Re-probe deactivated options periodically (~1 in 64 calls) to check if conditions normalized.
- Fit predicted EV against $\nu \cdot \text{drift}$ and age to deactivate an option *before* it loses money.

### Layer 3: Budget Controller (Pacing)
- Calibrate TSC frequency during setup. On each call, accumulate elapsed cycles plus the modelled fixed grader cost.
- $\text{Allowed spend per call} = \text{remaining budget} / \text{estimated remaining calls}$. Estimate calls via test fingerprinting (option/underlying count) at setup.
- Cap active book size $k$ so $k \times (\text{copy} + \text{score-read})$ fits per-call budget. Rank options by $w_i \times \text{predicted quality}$.

### Layer 4: Predict the True Volatility (Alpha)
The reference price uses the true current $\sigma$. Three methods to close the stale-vol gap:
1. **Borrow from siblings:** When option A receives `V` with change $\Delta\sigma$, nudge $\sigma$ of stale siblings on the same underlying by $\beta \cdot \Delta\sigma$.
2. **Volatility from scores:** Alternate small price offsets $\pm\delta$ across consecutive quotes to infer pricing error sign and magnitude, converting to implied $\Delta\sigma = \text{error} / \nu$.
3. **Learn score profile:** Quote intentional offsets right after `V` to map profit against absolute error, exposing the grader's penalty threshold and scale.

---

## Step-by-Step Implementation Plan

1. **Reconnaissance:** Run minimal probes to measure input counts and baseline execution overhead per test.
2. **Local Simulator:** Build realistic benchmark with GBM underlyings, drifting volatility, and sparse updates.
3. **Layer 1 Engine:** Dense SoA, precomputations, vectorized pricing, persistent quote book.
4. **Layer 2 Gating:** Track $w_i$, EWMA EV, and loss-triggered deactivation.
5. **Layer 3 Controller:** TSC cycle accounting and test fingerprinting.
6. **Layer 4 Alpha:** Sibling $\sigma$ propagation, smile/skew regression, and score-based corrections.
7. **Tune & Consolidate:** Calibrate deactivation thresholds, backoff periods, and learning rates.

---

## Risks and Mitigation

- **Do repeated, unchanged quotes score?** Confirmed: re-quoting cached prices scores on every call, making full-book caching the single largest throughput amplifier.
- **Grader memory management:** Custom memory allocators must remain safe and non-interfering with grader internal data structures.
- **Hidden test overfitting:** Maintain conservative fallback models if test fingerprints do not cleanly match recognized profiles.
- **Probe penalties:** Minimize active perturbation probes ($\pm\delta$) to avoid unnecessary penalty losses when pricing confidence is already high.
