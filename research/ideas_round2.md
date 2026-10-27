# Research ideas, round 2

New facts: J (toxic-wake filter) and K (J + stale-V lag anchor) both scored exactly H: ~90.27, test 5 = 4.90.
Every slope-based model since E gives test 5 = 4.89-4.91, whatever its form (additive, log, curvature, drift,
smile, smoothing, lag, toxic filter). Something structural caps test 5, so measure the headroom before modelling more.

## Why the vol model is probably not the bottleneck on test 5
- Realized move per input / sigma (x) gives each test's time step if spot is GBM with roughly the quoted vol:
  sqrt(dt) ~ x, so dt ~ 2.4e-5 yr/input (test 1), 9.6e-8 (test 5, ~3 s), ~2e-10 (test 6, ~6 ms if sigma ~0.3).
  In test 5 spot moves only ~3.5e-4 between an option's V updates (~12 calls apart), yet ~26% of V cycles end in a loss.
  So vol changes between V's are mostly NOT explained by spot, lag, noise or drift (all tested). Either they are
  unpredictable (then only gating matters), V is biased against the reference, or test 5 is simply worth little.

## D. Headroom diagnostics (Track B telemetry). One number per test per submission, spin at the last input
   (input counts are known: 2000/10069/51519/299752/139873/1e6). Use coarse buckets (e.g. 5 ms per percent) so
   timing noise doesn't matter.
D1 Profit efficiency: sum(profit) / sum over quotes of pmax_i, where pmax_i = max profit ever seen for option i.
   High on test 5 (>=80%) => test 5 is just low weight; stop working on it and put everything into test 6.
D2 Age-0 ratio: mean profit of quotes sent in the same call as the option's V, divided by pmax_i.
   Well below 1 on test 5 => the V itself is off from the reference (bias or staleness the lag model can't see).
D3 Loss share: sum(|losses|) / sum(gains). Large => gating is where the points are.
D4 Coverage: scores / (inputs x options that have a V), plus registered option count for tests 1-5
   (extend diag14's 4 ms/option to every test). Low => options are benched too long (backoff 64 too slow), or some
   options never get a V and could be priced from the smile fitted on their siblings.

## E. Model experiments (Track A), only where diagnostics point
E1 Clamp check: E/H/K clamp the slope at |beta| <= 5 (additive sigma per unit lnS). Test 5's slopes may be huge,
   because spot barely moves yet the slope gave +2.04. One-line test: H with LLC_BETA_MAX 50.
E2 If D2 is low: learn a per-option sigma offset by alternating whole V cycles at +delta / -delta and comparing the
   mean age-0..k profit of each side (Kiefer-Wolfowitz stochastic approximation with paired cycles). Each cycle
   starts from a fresh V, so the comparison is clean. Test 5 has ~42k cycles, so the A/B converges fast.
   Different from the failed per-tick probing: it runs at cycle level and starts small (delta = 0.25 x band/vega).
E3 If D3 is high: per-option contextual bandit for the quote/skip decision. Context = calls since V and |dlnS|
   since V (bucketed). Keep the mean profit per bucket and quote only where the lower confidence bound is
   above 0 (discounted UCB, Garivier & Moulines 2011). This replaces the bench-on-loss heuristic with learned
   expectations. Different from the failed profit-decay cutoff: it stops only where expected profit < 0.
E4 If D4 shows low coverage: shorter backoff (8 calls) and wake on a profit-floor instead of fixed time.

## Test 6 (unchanged priority, now the main upside)
With dt ~ 2e-10 yr per input, the reference vol barely moves across the whole test, so once found, a fixed sigma
earns for ~1M calls. Keep the sweep plan (v10_sweep / diag16) and the loss-shape check (ratio of max to min mean
|loss| across the stage-1 grid).
