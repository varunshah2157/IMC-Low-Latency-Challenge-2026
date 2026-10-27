# Research round 4: test 6 tracker spec from diag21-23

## My decode (baseline test 6 ~168.9 ms; formulas from the diag source)
- diag22 (714.1 ms): +545 = 450 (a quote MADE MONEY) + 95 => 190 of the 801 fine-scan points (0.02% steps,
  +-8% around sigma ~1.1) beat the capped loss. So the "beats the cap" zone is ~3.8% wide in sigma (+-1.9%),
  and a profitable apex exists inside it.
- diag23 (budget reached): 168.9 + 450 + 400*depth > 1000 => depth > ~0.95, consistent with reaching break-even+.
- diag21 (370.9 ms): +202 = 200 (flat) + ~2 => the SECOND sweep at call 600,000 found NOTHING beating the cap
  anywhere in [0.005, 2.5] at 0.5% steps. A 3.8%-wide zone would have been hit ~7 times if it were in range.
  => by call 600k the reference sigma had left [0.005, 2.5] (or moved a lot). Test 6 sigma drifts.
  If it is a random walk in log sigma from ~1.1, leaving the range by 600k needs a per-call std of at least
  ~0.82/sqrt(600000) ~ 0.1% per call. Compare the zone half-width 1.9%: tracking is feasible.
- diag16 cost of a whole sweep: about -0.07 score. Capped losses are tiny, so searching is cheap; missing
  the band is what costs.

## Tracker spec (v11 / final)
1. Acquire (as now): coarse 40 pts, then 0.5%, then the +-8% fine scan at 0.02%. Widen the range to
   [0.001, 20], since sigma may wander above 2.5.
2. Calibrate the V shape from the fine scan: fit p = a - b*|ln(sigma) - m| above the cap, by least squares over
   the points that beat the cap (fit m, a, b; check both flanks have the same slope).
   This gives |e| = (a - p)/b from any later profit above the cap.
3. Track, every call, with a two-hypothesis Kalman filter (Gaussian-sum; Alspach & Sorenson 1972):
   state m (log sigma of the reference), random-walk variance q per call (start q = (0.1%)^2, learn it from
   the innovations). Measurement: |m - x_quoted| = (a - p)/b, noise small. Two branches for the sign, each
   reweighted by its likelihood. Quote x = mean of the heavier branch. If the branches are close in weight,
   quote between them biased to the heavier one; the next profit splits them.
   Cheaper alternative: extremum-seeking dither (+-0.05% on alternate calls, demodulate, step).
4. Losing lock: profit at the cap for 3 calls => outward search around the last m: m +- 3%, +-6%, +-9%, ...
   alternating sides (spiral search), never more than the full range. Expected re-acquire cost is tiny.
5. Keep the sign test cheap: one wrong-side quote costs at most the cap.

## Why it should pay
Test 6 runs 1M calls with one option. Even at a fraction of the per-quote gain of other tests, holding the
apex for most calls could be worth many points; nobody at 77.95 earns anything here.

## Test 5 link
The same filter is the test 5 between-V tracker (V gives an exact reset of m; the filter tracks until the next V).
