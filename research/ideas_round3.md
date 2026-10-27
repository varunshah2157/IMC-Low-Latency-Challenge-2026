# Research ideas, round 3

## What the latest results say
- L (slope cap 5 -> 50) = H exactly. The cap never binds; drop that line.
- diag14: test 6 has about 1 option (+4.8 ms at 4 ms per option) and no known V. diag15: no unknown underlyings.
- diag16 (decode of 503.9 ms vs 168.5 baseline): +200 "stage 1 flat" fired, no +400, so no quote ever earned.
  Remaining 135 ms = 25 x ln(sigma/0.005) => sigma_best ~ 1.1.
  "Flat" means all 40 coarse points (0.005..2.5, 17% apart) returned the SAME profit to 1e-12.
  => far from the reference the loss is a constant cap, which is exactly why every gradient search died.
  The 0.5% sweep then found one point near sigma ~1.1 that beat the cap but was still not positive,
  and the 0.05% refine around it never went positive either.
  Two readings: (a) the profitable band is narrower than our steps and we only clipped its sloped edge;
  (b) the reference sigma moves over time, so the point had moved by the time we refined.
  diag21 (drift), diag22 (width) and diag23 (depth) in Track B test exactly this.

## Test 6: acquire, then track (Track B)
T1 Fit the apex instead of hunting for a positive point. Inside the sloped zone, profit is probably
   linear in |price error| (profit = w - k|err|, capped at -L). Two sloped points on the same side give
   the line; the apex lies where profit reaches w, and the mirror point on the other side confirms it.
   So after any point beats the cap: scan outward both ways at fine price steps until the cap returns,
   fit a line to each flank, and jump to where they meet. That needs no luck landing inside a narrow band.
T2 Track with extremum-seeking control (ESC) once inside. Source: Krstic & Wang (2000), "Stability of
   extremum seeking feedback for general nonlinear dynamic systems", Automatica; Ariyur & Krstic (2003),
   "Real-Time Optimization by Extremum-Seeking Control". Add a tiny dither +-eps to the quoted price on
   alternate calls, multiply each profit by the dither sign, low-pass the result: that is the gradient
   sign. Move sigma by gain x gradient. It follows a drifting optimum using only scalar scores.
   eps should be ~10% of the measured band half-width (from T1), so the dither costs little.
T3 Re-acquire: if profit sits at the cap for k calls in a row, the lock is lost; re-sweep
   locally (+-2% in price around the last lock, widening x2 each failure) before a full sweep.
   This is the acquisition/tracking split used in phase-locked loops and radar trackers.
T4 Step size in price space: step_sigma = (band_price / 2) / vega(sigma). The 0.5% sigma step at
   sigma ~1.1 may be wider than the band; diag22's 0.02% fine scan will give band width directly.

## Tests 1-5: use profit as a distance meter (Track A, after diag17-20)
P1 If losses turn out to be a constant cap (diag19: loss size in pmax units clusters at one value),
   the score is roughly p = w(1 - |err|/thr) above the cap. Then each profit tells |err| exactly:
   |err|/thr = 1 - p/w, with w = the option's max profit (learned per option, e.g. the 95th percentile).
   Optimal stopping: estimate the per-call drift of |err| from successive profits (std s per call).
   Stop quoting when P(|err| next call > thr) x L > (1 - P) x E[gain], with P from a normal step of
   std s from the current |err|. This replaces bench-after-loss, which always pays one full loss L
   before reacting. Different from the failed decay cutoffs: those stopped at 25% or 50% of the first
   profit, regardless of the loss size and the drift speed.
P2 Lower priority: the same ESC dither between V updates to learn the sign of the error and nudge sigma.
   v1's probing failed, but it used large sigma steps without demodulation. Only try this if diag20 (age-0
   efficiency) shows quotes are already off the apex right after a V.

## Check with Saksham
What do the top score totals look like? If the leaders are ~95, the gap is test 5 and 6 polish;
if they are ~110+, someone has cracked test 6 and that is where all effort should go.

## Update: test 5 vol drifts fast (from diag_test5)
Only ~2% of quotes lose, but a quote 3+ calls after a V earns ~12% of the on-update profit (test 4: >99%).
So V is accurate on arrival and then sigma runs away within a few calls. Ideas to predict or track it:

Q1 Profit inversion tracker (strongest; also the tracker for test 6 after acquisition).
   Step 1, calibrate the score curve: right after a V (error ~0), on a small share of cycles quote
   sigma_V shifted by a known price offset d (grid of +-d values in units of the option's typical move), and
   record profit p(d)/w. That maps the hidden score function f(|err|) and checks it is symmetric.
   A few hundred samples out of ~42k cycles is enough, so the cost is small.
   Step 2, invert: between V's, each profit gives |err| = f^-1(p/w).
   Step 3, resolve the sign with one move: shift the quote by +|err|. If the next profit returns to ~w,
   the sign was right; if it drops to f(2|err|), flip to -|err|. Keep re-inverting every call.
   This is Newton-like: it locks on in 1-2 calls, whereas gradient probing (v1) needed many calls and
   paid for every probe. It needs the reference to move less per call than the band, which the 12%
   figure (decay over ~3 calls) suggests.
Q2 Vol momentum: v7 only switched on for NEGATIVE autocorrelation of V-to-V changes (noise).
   Positive autocorrelation (trend) was never tested. Per option, velocity v = (sigma_k - sigma_k-1)/calls
   between them; predict sigma(t) = sigma_k + phi * v * (t - t_k), with phi fitted out of sample and
   selected per option in the ensemble. Also try the siblings' latest velocity (common trend).
Q3 Feed Q1/Q2 a sign prior: use momentum (Q2) or the siblings' latest V surprise as the first guess of
   the sign in Q1 step 3, so the first move is usually right.
