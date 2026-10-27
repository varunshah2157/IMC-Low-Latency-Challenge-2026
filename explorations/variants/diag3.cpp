// Diagnostic submission: never quotes (scores 0), and burns a known amount of
// time per counted event so "Time" on the results page reveals the count.
//   time_ms ~= baseline (about 0.12-0.17 us per input) + count * SPIN_US / 1000
#include "api.hpp"
#include <chrono>
#include <unordered_set>

#ifndef DIAG_MODE
#define DIAG_MODE 3
#endif
// 1: count V updates whose id is a registered option        (2 us each)
// 2: count updates (U or V) whose id is not registered      (20 us each, max 30000)
// 3: count registered options, spent on the first call      (10 us each)

static void spinUs(double us) {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {}
}

class Solution : public ISolution {
    std::unordered_set<int> opts, unds;
    long calls = 0, bad = 0;
public:
    void registerOption(const Option& o) override { opts.insert(o.id); unds.insert(o.underlyingId); }
    std::vector<Valuation> process(const ParameterUpdate& u, const std::vector<Score>&) override {
        ++calls;
#if DIAG_MODE == 1
        if (u.type == 'V' && opts.count(u.instrumentId)) spinUs(2.0);
#elif DIAG_MODE == 2
        bool known = u.type == 'V' ? opts.count(u.instrumentId) : (u.type == 'U' && unds.count(u.instrumentId));
        if (!known && bad < 30000) { ++bad; spinUs(20.0); }
#else
        if (calls == 1) spinUs(10.0 * double(opts.size()));
#endif
        return {};
    }
};

ISolution* createSolution() { return new Solution(); }
