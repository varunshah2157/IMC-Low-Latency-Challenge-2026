// Local simulator for the Low Latency Challenge.
//
// Builds a synthetic market (GBM underlyings, random-walk volatilities that are only
// revealed by sparse V updates), scores valuations against the Black-Scholes price at
// the TRUE current volatility, and times process() the way the grader does (sum of
// in-call time plus a fixed 0.1 us per call).
//
// The real scoring function is hidden; this one is "weight * (1 - |err| / threshold)",
// which has the documented shape: closer is better, past the threshold you lose money.
//
// Usage: ./sim [seed] [n_under] [opts_per_under] [n_updates] [p_vol] [vol_sd] [thr]

#include "api.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <x86intrin.h>

static double bs_exact(double S, double K, double T, double sig, double r = 0.04) {
    const double sv = sig * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / sv;
    const double d2 = d1 - sv;
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

int main(int argc, char** argv) {
    const unsigned seed = argc > 1 ? atoi(argv[1]) : 1;
    const int n_under = argc > 2 ? atoi(argv[2]) : 10;
    const int per_under = argc > 3 ? atoi(argv[3]) : 20;
    const long n_updates = argc > 4 ? atol(argv[4]) : 200000;
    const double p_vol = argc > 5 ? atof(argv[5]) : 0.05;
    const double vol_sd = argc > 6 ? atof(argv[6]) : 0.002;  // vol random-walk sd per tick
    const double thr = argc > 7 ? atof(argv[7]) : 0.05;      // absolute price error where profit hits 0
    const double frac_novol = argc > 8 ? atof(argv[8]) : 0.0;  // share of options that never get a V update
    const int loss_shape = argc > 9 ? atoi(argv[9]) : 0;       // 0 linear, 1 loss capped at -w, 2 quadratic
    const double beta_true = argc > 10 ? atof(argv[10]) : 0.0;  // true d(sigma)/d(ln S): vol moves with spot
    const double smile = argc > 11 ? atof(argv[11]) : 0.0;      // per-underlying smile sigma += b x + c x^2, x = ln(S/K)
    const double vnoise = argc > 12 ? atof(argv[12]) : 0.0;     // sd of noise on each reported vol
    const double frac_toxic = argc > 13 ? atof(argv[13]) : 0.0; // share of options whose reported vol is 25% off
    const int vlag = argc > 14 ? atoi(argv[14]) : 0;             // reported vol is for the spot vlag updates ago
    const double cf_sd = argc > 15 ? atof(argv[15]) : 0.0;       // common vol factor per underlying, sd per tick
    const double spot_sd = argc > 16 ? atof(argv[16]) : 0.001;   // spot log move per U update
    std::vector<std::vector<double>> shist(n_under);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    struct Opt { int id, u; double K, T, w, sig, known_sig; long last_t; bool has_known, novol, toxic; };
    std::vector<double> S(n_under);
    std::vector<int> uid(n_under);
    std::vector<Opt> opts;
    for (int u = 0; u < n_under; ++u) {
        S[u] = 50 + 150 * unif(rng);
        uid[u] = 1000 + 37 * u;
        for (int j = 0; j < per_under; ++j) {
            Opt o;
            o.id = 500000 + (int)opts.size() * 13;
            o.u = u;
            o.K = std::round(S[u] * (0.7 + 0.6 * unif(rng)));
            o.T = 0.02 + 1.0 * unif(rng);
            o.w = std::exp(1.5 * gauss(rng));  // some options carry far more money
            o.sig = 0.15 + 0.5 * unif(rng);
            o.known_sig = 0;
            o.last_t = 0;
            o.has_known = false;
            o.novol = unif(rng) < frac_novol;
            o.toxic = unif(rng) < frac_toxic;
            opts.push_back(o);
        }
    }
    std::vector<int> idx_of_id(500000 + opts.size() * 13 + 1, -1);
    for (size_t i = 0; i < opts.size(); ++i) idx_of_id[opts[i].id] = (int)i;

    // advance an option's true vol lazily (Brownian increments compose exactly)
    // True vol path: a random walk on a fixed grid of 32 ticks, linearly interpolated, drawn
    // from a per-option RNG so the market does not depend on what the solution quotes.
    std::vector<double> S0, sb(n_under), sc(n_under);
    for (int u = 0; u < n_under; ++u) sb[u] = -1.5 + 2.0 * unif(rng), sc[u] = 3.0 * unif(rng);
    std::vector<std::vector<double>> vgrid(opts.size());
    std::vector<std::mt19937_64> vrng, frng;
    std::vector<std::vector<double>> fgrid(n_under);
    for (int u = 0; u < n_under; ++u) frng.emplace_back(seed * 7919ull + u);
    for (size_t i = 0; i < opts.size(); ++i) vrng.emplace_back(seed * 1000003ull + i);
    auto true_sig = [&](Opt& o, long t) {
        const size_t i = &o - &opts[0];
        auto& g = vgrid[i];
        const size_t need = t / 32 + 2;
        if (g.empty()) g.push_back(o.sig);
        while (g.size() < need) {
            double v = g.back() + vol_sd * std::sqrt(32.0) * gauss(vrng[i]);
            if (v < 0.05) v = 0.1 - v;
            g.push_back(v);
        }
        const double f = (t % 32) / 32.0;
        const double base = g[t / 32] * (1 - f) + g[t / 32 + 1] * f;
        const double x = std::log(S[o.u] / o.K), x0 = std::log(S0[o.u] / o.K);
        const double sm = smile * (sb[o.u] * (x - x0) + sc[o.u] * (x * x - x0 * x0));
        double fac = 0.0;
        if (cf_sd > 0) {
            auto& fg = fgrid[o.u];
            const size_t need2 = t / 32 + 2;
            if (fg.empty()) fg.push_back(0.0);
            while (fg.size() < need2) fg.push_back(fg.back() + cf_sd * std::sqrt(32.0) * gauss(frng[o.u]));
            fac = (fg[t / 32] * (1 - f) + fg[t / 32 + 1] * f) * (0.5 + (i % 3) * 0.5);
        }
        const double s = base + beta_true * std::log(S[o.u] / S0[o.u]) + sm + fac;
        return s < 0.02 ? 0.02 : s;
    };
    S0 = S;
    ISolution* sol = createSolution();
    for (auto& o : opts) sol->registerOption(Option{o.id, uid[o.u], (int)o.K, o.T});

    double tsc_hz;
    {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t c0 = __rdtsc();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.05) {}
        tsc_hz = (__rdtsc() - c0) / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    std::vector<Score> prev;
    std::vector<long> scored_at(opts.size(), -1);
    double total = 0, used_sec = 0, max_engine_err = 0;
    long n_vals = 0, n_neg = 0, processed = 0;
    for (long t = 0; t < n_updates; ++t) {
        ParameterUpdate up;
        Opt* vo = &opts[rng() % opts.size()];
        if (unif(rng) < p_vol && !vo->novol) {
            Opt& o = *vo;
            const double s_now = S[o.u];
            if (vlag > 0 && !shist[o.u].empty())
                S[o.u] = shist[o.u][shist[o.u].size() > (size_t)vlag ? shist[o.u].size() - 1 - vlag : 0];
            const double s = std::max(0.01, true_sig(o, t) * (o.toxic ? 1.25 : 1.0) + vnoise * gauss(rng));
            S[o.u] = s_now;
            o.known_sig = s;
            o.has_known = true;
            up = ParameterUpdate{'V', s, o.id};
        } else {
            const int u = rng() % n_under;
            S[u] *= std::exp(spot_sd * gauss(rng));
            shist[u].push_back(S[u]);
            up = ParameterUpdate{'U', S[u], uid[u]};
        }
        const uint64_t c0 = __rdtsc();
        std::vector<Valuation> vals;
        try {
            vals = sol->process(up, prev);
        } catch (...) {
            std::printf("crashed on input %ld\n", t + 1);
            break;
        }
        const uint64_t c1 = __rdtsc();
        used_sec += (c1 - c0) / tsc_hz + 0.1e-6;
        if (used_sec > 1.0) break;
        processed = t + 1;

        prev.clear();
        const size_t cap = opts.size() + 64;
        for (size_t j = 0; j < vals.size() && j < cap; ++j) {
            const auto& v = vals[j];
            if (v.optionId < 0 || v.optionId >= (int)idx_of_id.size()) continue;
            const int i = idx_of_id[v.optionId];
            if (i < 0 || !std::isfinite(v.price) || scored_at[i] == t) continue;
            scored_at[i] = t;
            Opt& o = opts[i];
            const double ref = bs_exact(S[o.u], o.K, o.T, true_sig(o, t));
            const double e = std::fabs(v.price - ref) / thr;
            double profit = loss_shape == 2 ? o.w * (1.0 - e * e) : o.w * (1.0 - e);
            if (loss_shape == 1 && profit < -o.w) profit = -o.w;
            if (o.has_known) {
                const double eng = std::fabs(v.price - bs_exact(S[o.u], o.K, o.T, o.known_sig));
                if (eng > max_engine_err) max_engine_err = eng;
            }
            prev.push_back(Score{v.optionId, profit});
            total += profit;
            ++n_vals;
            n_neg += profit < 0;
        }
    }
    std::printf("score %12.1f | processed %ld/%ld | time %.3f s | vals %ld (%.1f/call) | losing %.1f%% | max engine err %.2e\n",
                total, processed, n_updates, used_sec, n_vals, (double)n_vals / std::max(1L, processed),
                100.0 * n_neg / std::max(1L, n_vals), max_engine_err);
    return 0;
}
