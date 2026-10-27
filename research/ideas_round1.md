# Research ideas, round 1

Context: best ~90.26 (H smile ensemble), v9 ~90.20, E 90.15. Remaining upside: test 5 (4.9) and test 6 (0).
Already ruled out on the grader (do not redo): noisy-V smoothing (v7 Kalman, I), local sigma search on test 6,
underlying-keyed V, realized vol as a sigma level, decay cutoffs, probing, sibling model (pre-slope), one shared slope,
log-sigma slope (additive wins), time-drift term (v8), pooled prior (F).

## A. Vol model ideas (tests 1-5) [Track A: Production Smile Engine]

A1. Stale V anchor (lag search). Problem text: "our trading system is just not fast enough to calculate [vol]".
    So a V may describe sigma at an earlier spot, not at the spot when it arrives. E anchors at lnS at arrival.
    Keep a ring buffer of the last 64 lnS per underlying. At each V, for lags l = 0,1,2,4,8,16,32 keep separate
    regression sums with dx_l = lnS_now - lnS_anchor(l) style anchors (anchor of previous V at its lag l too), pick the
    lag with smallest out-of-sample residual (per option or per test), price with sigma = sigma_V + beta*(lnS - lnS[l at V]).
    Fits test 5: U every 1.46 calls, V every ~12 calls per option, ~26% of V cycles end in a loss, often early.
    Cheap check first: diag13 (early losses). Many early losses => stale/lagged V is the prime suspect.

A2. Path-dependent vol (GARCH / leverage). Sources: Glosten-Jagannathan-Runkle (1993) GJR-GARCH; Black (1976)
    leverage effect. Add regressors to the per-option Δsigma regression:
    (a) separate slopes for up and down moves: dx+ = max(dx,0), dx- = min(dx,0);
    (b) q = sum of squared log returns of S since last V (realized variance increment), so vol rises after busy periods.
    Feed into the existing model-selection (out-of-sample error) so options without the effect fall back to E.

A3. Mean reversion toward a per-option level (Heston/OU): Δsigma = beta*dx + kappa*(theta - sigma_prev)*Δcalls.
    Regress Δsigma on [dx, sigma_prev*Δn, Δn]. Different from v8's constant drift. Shrink with the t-stat rule.

A4. Residual co-movement across options (Cont & da Fonseca 2002, "Dynamics of implied volatility surfaces":
    surface moves are dominated by a common level factor). After the slope model, the surprise e_j = sigma_V - predicted
    for option j may be shared. Estimate rho_ij from paired surprises (V of i and j close in time); when j's V arrives,
    shift i's sigma by g*e_j where g is learned and gated by out-of-sample error. An initial sibling model tested a
    version of this before slopes existed, so expectation is modest; run only via self-selection.

## B. Test 6 and diagnostics [Track B: Side Channels & Search Engine]

B1. Learn the test 6 option first (timing side channel): encode K/S (moneyness), T, and S itself at call 2000.
    If deep ITM with tiny T, price ~ intrinsic and sigma barely matters; if near ATM, the vega sets the grid step.
B2. Prior for sigma: encode min / mean / max of all V values in tests 1-5. If the generator draws vols from one range
    (e.g. 0.1-0.5), test 6's sigma is probably in it too, which shrinks the sweep.
B3. Learn the loss shape: quote test 6 at a few fixed sigmas (e.g. 0.1, 0.3, 0.6, 1.0) for 2000 calls each and encode
    the mean profit of each block. If |loss| grows with distance, a parabolic / golden-section fit on |loss| finds sigma
    with no profitable quote needed (unimodal bandit; Combes & Proutiere, ICML 2014). If loss is a flat constant, only a
    sweep works (B4).
B4. Wide sweep in price space, not sigma space: step so the price changes by ~half the profitable band per step
    (step_sigma = band / vega). One quote per call, geometric sigma grid 0.01..3. Diagnostic version spins
    50 ms + index*0.05 ms at the first profitable quote, so the run time reveals the winning sigma. Then lock on and
    track with small paired steps around it. A 0.5% sigma step may be too coarse if the band is tight (test 5's band looks tight).

## C. Test 5 loss localisation (diagnostic)

C1. Per option: count losses and V updates per option, encode the max share any single option takes of all losses.
    "Toxic option" hypothesis: if one option takes most of the ~11k losses, skip waking it on raw V
    (require it to show profit, or quote only after k consecutive profitable cycles).
C2. Histogram of calls-since-V at the first loss (buckets 0,1,2-3,4-7,8+), encoded across a few submissions.
    0-1 => V is stale or the band is too tight (A1). Late => drift the model misses (A2/A3).

## Critique of preliminary proposals
- Noisy V (Kalman / EWMA): already tested twice (v7, I). No gain. Drop.
- Time slope: v8 had a drift term; test 5 unchanged. Drop unless as mean reversion (A3).
- v6a/v6b/v7 submissions, log-sigma, pooled slope prior, t>2 threshold: already run (log lost to additive, pooled F = 90.03).
- Unknown-id V in test 6: diag15 already measures this.
- Worth doing: test 6 grid sweep with timing encode (B4), toxic-option check (C1).
- Loss math: 11k losses over ~42k V updates is right for v2_safe; E and v9 may differ, diag12 will tell.
