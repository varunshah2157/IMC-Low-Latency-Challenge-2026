#include "api.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <vector>
#include <x86intrin.h>
#include <immintrin.h>
#include <malloc.h>
namespace hS {
bool g_mode0 = false;
#ifndef LLC_REQUOTE_ALL
#define LLC_REQUOTE_ALL 1
#endif
#ifndef LLC_GATING
#define LLC_GATING 1
#endif
#ifndef LLC_NEG_K
#define LLC_NEG_K 1
#endif
#ifndef LLC_BACKOFF0
#define LLC_BACKOFF0 256
#endif
#ifndef LLC_BACKOFF_MAX
#define LLC_BACKOFF_MAX 65536
#endif
#ifndef LLC_BUDGET_SEC
#define LLC_BUDGET_SEC 0.92
#endif
#ifndef LLC_EXT_NS_CALL
#define LLC_EXT_NS_CALL 150.0
#endif
#ifndef LLC_EXT_NS_VAL
#define LLC_EXT_NS_VAL 0.5
#endif
#ifndef LLC_BETA
#define LLC_BETA 1
#endif
#ifndef LLC_BETA_FORGET
#define LLC_BETA_FORGET 0.99
#endif
#ifndef LLC_BETA_MODE
#define LLC_BETA_MODE 3
#endif
#ifndef LLC_BETA_FORGET_GLOBAL
#define LLC_BETA_FORGET_GLOBAL 0.9995
#endif
#ifndef LLC_SMILE_FORGET_U
#define LLC_SMILE_FORGET_U 0.999
#endif
#ifndef LLC_ENS_FORGET
#define LLC_ENS_FORGET 0.99
#endif
#ifndef LLC_ENS_MARGIN
#define LLC_ENS_MARGIN 0.97
#endif
#ifndef LLC_BETA2_MAX
#define LLC_BETA2_MAX 50.0
#endif
#ifndef LLC_BETA_MAX
#define LLC_BETA_MAX 5.0
#endif
#ifndef LLC_SEARCH
#define LLC_SEARCH 0
#endif
#ifndef LLC_SEARCH_START
#define LLC_SEARCH_START 1000
#endif
#ifndef LLC_SEARCH_BLOCK
#define LLC_SEARCH_BLOCK 4
#endif
#ifndef LLC_SEARCH_SIGMA0
#define LLC_SEARCH_SIGMA0 0.3
#endif
#ifndef LLC_SEARCH_H0
#define LLC_SEARCH_H0 0.3
#endif
#ifndef LLC_SEARCH_HMIN
#define LLC_SEARCH_HMIN 2e-4
#endif
#ifndef LLC_SEARCH_VOL_SHARE_INV
#define LLC_SEARCH_VOL_SHARE_INV 200
#endif
#ifndef LLC_SEARCH_GIVEUP
#define LLC_SEARCH_GIVEUP 400
#endif
#ifndef LLC_SCORE_STRIDE
#define LLC_SCORE_STRIDE 4
#endif
#ifndef LLC_N_MAX
#define LLC_N_MAX 1000000
#endif
#ifndef LLC_N_GUESS
#define LLC_N_GUESS 0
#endif
namespace {
constexpr double kRate = 0.04;
static inline __attribute__((always_inline)) double fast_exp(double y) {
y = y < -708.0 ? -708.0 : y;
y = y > 709.0 ? 709.0 : y;
const double shifter = 6755399441055744.0;
const double kd = y * 1.4426950408889634 + shifter;
const double k = kd - shifter;
const double r = (y - k * 0.6931471803691238) - k * 1.9082149292705877e-10;
double p = 1.0 / 3628800.0;
p = p * r + 1.0 / 362880.0;
p = p * r + 1.0 / 40320.0;
p = p * r + 1.0 / 5040.0;
p = p * r + 1.0 / 720.0;
p = p * r + 1.0 / 120.0;
p = p * r + 1.0 / 24.0;
p = p * r + 1.0 / 6.0;
p = p * r + 0.5;
p = p * r + 1.0;
p = p * r + 1.0;
uint64_t bits;
std::memcpy(&bits, &kd, 8);
bits = (bits + 1023) << 52;
double scale;
std::memcpy(&scale, &bits, 8);
return p * scale;
}
static inline __attribute__((always_inline)) void norm_cdf_parts(double x, double& num, double& den) {
double a = x < 0.0 ? -x : x;
a = a > 38.0 ? 38.0 : a;
double n = 3.52624965998911e-02;
n = n * a + 0.700383064443688;
n = n * a + 6.37396220353165;
n = n * a + 33.912866078383;
n = n * a + 112.079291497871;
n = n * a + 221.213596169931;
n = n * a + 220.206867912376;
double d = 8.83883476483184e-02;
d = d * a + 1.75566716318264;
d = d * a + 16.064177579207;
d = d * a + 86.7807322029461;
d = d * a + 296.564248779674;
d = d * a + 637.333633378831;
d = d * a + 793.826512519948;
d = d * a + 440.413735824752;
num = fast_exp(-0.5 * a * a) * n;
den = d;
}
static inline __attribute__((always_inline)) double norm_cdf(double x) {
double n, d;
norm_cdf_parts(x, n, d);
const double tail = n / d;
return x > 0.0 ? 1.0 - tail : tail;
}
struct Kernel {
const double* a;
const double* sqrtT;
const double* sigv;
const double* beta;
const double* beta2;
const double* xv;
const double* kdisc;
double* price;
};
static inline __attribute__((always_inline)) void price_range_body(const Kernel& kn, int lo, int hi,
double S, double lnS) {
const double* __restrict a = kn.a;
const double* __restrict sqrtT = kn.sqrtT;
const double* __restrict sigv = kn.sigv;
const double* __restrict beta = kn.beta;
const double* __restrict beta2 = kn.beta2;
const double* __restrict xv = kn.xv;
const double* __restrict kdisc = kn.kdisc;
double* __restrict price = kn.price;
for (int i = lo; i < hi; ++i) {
const double dxs = lnS - xv[i];
double sig = sigv[i] + dxs * (beta[i] + beta2[i] * dxs);
sig = sig < 1e-4 ? 1e-4 : sig;
const double sv = sig * sqrtT[i];
const double d1 = (lnS + a[i]) / sv + 0.5 * sv;
const double d2 = d1 - sv;
double n1, e1, n2, e2;
norm_cdf_parts(d1, n1, e1);
norm_cdf_parts(d2, n2, e2);
const double inv = 1.0 / (e1 * e2);
const double t1 = n1 * e2 * inv, t2 = n2 * e1 * inv;
const double N1 = d1 > 0.0 ? 1.0 - t1 : t1;
const double N2 = d2 > 0.0 ? 1.0 - t2 : t2;
double p = S * N1 - kdisc[i] * N2;
const double intrinsic = S - kdisc[i];
p = p < intrinsic ? intrinsic : p;
p = p < 0.0 ? 0.0 : p;
price[i] = p;
}
}
__attribute__((target("avx512f,avx512dq,avx512vl,avx2,fma,prefer-vector-width=512"),
optimize("no-trapping-math,finite-math-only,no-math-errno"))) void price_range_avx512(const Kernel& kn, int lo, int hi, double S,
double lnS) {
price_range_body(kn, lo, hi, S, lnS);
}
__attribute__((target("avx2,fma"), optimize("no-trapping-math,finite-math-only,no-math-errno"))) void price_range_avx2(const Kernel& kn, int lo, int hi, double S,
double lnS) {
price_range_body(kn, lo, hi, S, lnS);
}
__attribute__((optimize("no-trapping-math,finite-math-only,no-math-errno"))) void price_range_generic(const Kernel& kn, int lo, int hi, double S, double lnS) {
price_range_body(kn, lo, hi, S, lnS);
}
struct IdMap {
bool direct = true;
int64_t base = 0;
std::vector<int32_t> table;
std::vector<int32_t> keys;
uint64_t mask = 0;
void build(const std::vector<std::pair<int, int>>& entries) {
if (entries.empty()) {
direct = true;
base = 0;
table.assign(1, -1);
return;
}
int64_t lo = entries[0].first, hi = lo;
for (auto& e : entries) {
lo = std::min<int64_t>(lo, e.first);
hi = std::max<int64_t>(hi, e.first);
}
const int64_t span = hi - lo + 1;
if (span <= (int64_t)entries.size() * 8 + (1 << 20)) {
direct = true;
base = lo;
table.assign((size_t)span, -1);
for (auto& e : entries) table[(size_t)(e.first - lo)] = e.second;
} else {
direct = false;
uint64_t cap = 16;
while (cap < entries.size() * 4) cap <<= 1;
mask = cap - 1;
keys.assign(cap, 0);
table.assign(cap, -1);
for (auto& e : entries) {
uint64_t h = hash(e.first);
while (table[h] != -1) h = (h + 1) & mask;
keys[h] = e.first;
table[h] = e.second;
}
}
}
inline uint64_t hash(int k) const {
return ((uint64_t)(uint32_t)k * 0x9E3779B97F4A7C15ull >> 20) & mask;
}
inline int find(int k) const {
if (direct) {
const uint64_t off = (uint64_t)((int64_t)k - base);
return off < table.size() ? table[off] : -1;
}
uint64_t h = hash(k);
while (table[h] != -1) {
if (keys[h] == k) return table[h];
h = (h + 1) & mask;
}
return -1;
}
};
double calibrate_tsc_hz() {
using clk = std::chrono::steady_clock;
const auto t0 = clk::now();
const uint64_t c0 = __rdtsc();
while (std::chrono::duration<double>(clk::now() - t0).count() < 0.02) {
}
const auto t1 = clk::now();
const uint64_t c1 = __rdtsc();
const double hz = (double)(c1 - c0) / std::chrono::duration<double>(t1 - t0).count();
return (hz > 1e8 && hz < 1e10) ? hz : 3.0e9;
}
#ifdef LLC_PROFILE
}
uint64_t g_prof[4];
namespace {
#endif
enum : uint8_t { NOT_READY = 0, ACTIVE = 1, SLEEPING = 2 };
}
class Solution : public ISolution {
public:
Solution() {
tsc_hz_ = calibrate_tsc_hz();
use_avx2_ = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
use_avx512_ = use_avx2_ && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
__builtin_cpu_supports("avx512vl");
}
void registerOption(const Option& option) override { reg_.push_back(option); }
std::vector<Valuation> process(const ParameterUpdate& update,
const std::vector<Score>& previousScores) override {
const uint64_t t_start = __rdtsc();
#ifdef LLC_PROFILE
uint64_t t_prev = t_start;
#define PROF(n) do { uint64_t t_ = __rdtsc(); g_prof[n] += t_ - t_prev; t_prev = t_; } while (0)
#else
#define PROF(n) do {} while (0)
#endif
if (!ready_) finalize();
++calls_;
#if LLC_GATING
if (!previousScores.empty() && (klimit_ > n_ || calls_ % LLC_SCORE_STRIDE == 0))
read_scores(previousScores);
wake_sleepers();
#endif
#if LLC_SEARCH
if (n_search_ > 0) {
if (!previousScores.empty()) collect_search_scores(previousScores);
if ((calls_ & 16383) == 0) check_search_pays();
if (calls_ % search_block_ == 0) search_step();
}
if (calls_ >= LLC_SEARCH_START && (calls_ & 1023) == (LLC_SEARCH_START & 1023)) start_search();
#endif
PROF(0);
n_changed_ = 0;
#if LLC_BETA && LLC_BETA_MODE == 2
if ((calls_ & 1023) == 0) refresh_global_beta();
#endif
if (update.type == 'U') {
on_underlying(update.instrumentId, update.value);
} else if (update.type == 'V') {
on_vol(update.instrumentId, update.value);
}
PROF(1);
#if LLC_REQUOTE_ALL
int k = nbook_;
if (k > klimit_) {
promote_changed();
k = klimit_;
}
#else
int k = 0;
for (int c = 0; c < n_changed_; ++c) {
const int i = changed_[c];
if (state_[i] == ACTIVE) swap_book(k++, pos_[i]);
}
#endif
std::vector<Valuation> out(book_.data(), book_.data() + k);
last_k_ = k;
last_side_ = search_side();
PROF(2);
const uint64_t t_end = __rdtsc();
account(t_end - t_start, k);
return out;
}
private:
void finalize() {
ready_ = true;
n_ = (int)reg_.size();
std::unordered_map<int, int> under_of;
std::vector<int> under_ids;
for (auto& o : reg_) {
if (under_of.emplace(o.underlyingId, (int)under_ids.size()).second) under_ids.push_back(o.underlyingId);
}
nu_ = (int)under_ids.size();
std::vector<int> cnt(nu_ + 1, 0);
for (auto& o : reg_) cnt[under_of[o.underlyingId] + 1]++;
u_start_.assign(nu_ + 1, 0);
for (int u = 0; u < nu_; ++u) u_start_[u + 1] = u_start_[u] + cnt[u + 1];
std::vector<int> fill(u_start_.begin(), u_start_.end() - 1);
opt_id_.resize(n_);
T_.resize(n_);
lnK_.resize(n_);
sqrtT_.resize(n_);
a_.assign(n_, 0.0);
sigv_.assign(n_, 1e-4);
beta_.assign(n_, 0.0);
beta2_.assign(n_, 0.0);
om_.assign(n_, OptModel{});
um_.assign(nu_, UndModel{});
xv_.assign(n_, 0.0);
bxx_.assign(n_, 0.0);
bxy_.assign(n_, 0.0);
byy_.assign(n_, 0.0);
bn_.assign(n_, 0.0);
bseen_.assign(n_, 0);
kdisc_.resize(n_);
price_.assign(n_, 0.0);
has_vol_.assign(n_, 0);
state_.assign(n_, NOT_READY);
pos_.assign(n_, -1);
under_.resize(n_);
neg_.assign(n_, 0);
backoff_.assign(n_, LLC_BACKOFF0);
gen_.assign(n_, 0);
ev_.assign(n_, 0.0);
searching_.assign(n_, 0);
dir_.assign(n_, 0);
pairs_.assign(n_, 0);
ls_.assign(n_, 0.0);
h_.assign(n_, 0.0);
ssum_.assign(2 * n_, 0.0);
scnt_.assign(2 * n_, 0);
search_list_.assign(n_ + 1, 0);
book_.resize(n_ + 1);
book_idx_.resize(n_ + 1);
changed_.resize(n_ + 1);
to_sleep_.reserve(n_ + 1);
std::vector<std::pair<int, int>> oent;
oent.reserve(n_);
for (auto& o : reg_) {
const int u = under_of[o.underlyingId];
const int i = fill[u]++;
opt_id_[i] = o.id;
under_[i] = u;
const double T = o.tte > 1e-12 ? o.tte : 1e-12;
T_[i] = T;
sqrtT_[i] = std::sqrt(T);
lnK_[i] = std::log((double)o.strike);
kdisc_[i] = (double)o.strike * std::exp(-kRate * T);
a_[i] = kRate * T - lnK_[i];
oent.emplace_back(o.id, i);
}
std::vector<std::pair<int, int>> uniq;
{
std::unordered_map<int, int> seen;
for (auto& e : oent)
if (seen.emplace(e.first, e.second).second) uniq.push_back(e);
}
idmap_opt_.build(uniq);
std::vector<std::pair<int, int>> uent;
for (int u = 0; u < nu_; ++u) uent.emplace_back(under_ids[u], u);
idmap_und_.build(uent);
u_dirty_.assign(nu_, 0);
dirty_list_.assign(nu_ + 1, 0);
S_.assign(nu_, 0.0);
lnS_.assign(nu_, 0.0);
has_S_.assign(nu_, 0);
kernel_ = Kernel{a_.data(), sqrtT_.data(), sigv_.data(), beta_.data(), beta2_.data(), xv_.data(),
kdisc_.data(), price_.data()};
klimit_ = n_ + 1;
budget_cyc_ = LLC_BUDGET_SEC * tsc_hz_;
ext_call_cyc_ = LLC_EXT_NS_CALL * 1e-9 * tsc_hz_;
ext_val_cyc_ = LLC_EXT_NS_VAL * 1e-9 * tsc_hz_;
reg_.clear();
reg_.shrink_to_fit();
}
void price_range(int lo, int hi, double S, double lnS) {
if (use_avx512_)
price_range_avx512(kernel_, lo, hi, S, lnS);
else if (use_avx2_)
price_range_avx2(kernel_, lo, hi, S, lnS);
else
price_range_generic(kernel_, lo, hi, S, lnS);
}
void on_underlying(int uid, double S) {
const int u = idmap_und_.find(uid);
if (u < 0 || !(S > 0.0) || !std::isfinite(S)) return;
S_[u] = S;
const double lnS = std::log(S);
lnS_[u] = lnS;
const int lo = u_start_[u], hi = u_start_[u + 1];
if (!has_S_[u]) {
has_S_[u] = 1;
for (int i = lo; i < hi; ++i) xv_[i] = lnS;
}
price_range(lo, hi, S, lnS);
for (int i = lo; i < hi; ++i) {
const uint8_t st = state_[i];
if (st == ACTIVE) {
book_[pos_[i]].price = price_[i];
changed_[n_changed_++] = i;
} else if (st == NOT_READY && has_vol_[i]) {
activate(i);
changed_[n_changed_++] = i;
}
}
}
void on_vol(int oid, double sigma) {
++n_vol_updates_;
if (!(sigma >= 0.0) || !std::isfinite(sigma)) return;
const int i = idmap_opt_.find(oid);
if (i < 0) {
const int u = idmap_und_.find(oid);
if (u >= 0)
for (int j = u_start_[u]; j < u_start_[u + 1]; ++j) apply_vol(j, sigma);
return;
}
apply_vol(i, sigma);
}
void apply_vol(int i, double sigma) {
#if LLC_BETA
learn_beta(i, sigma);
#endif
const double b = beta_[i], b2 = beta2_[i];
set_sigma(i, sigma);
#if LLC_BETA
beta_[i] = searching_[i] ? 0.0 : b;
beta2_[i] = searching_[i] ? 0.0 : b2;
#endif
has_vol_[i] = 1;
searching_[i] = 0;
neg_[i] = 0;
backoff_[i] = LLC_BACKOFF0;
++gen_[i];
const int u = under_[i];
if (!has_S_[u]) return;
price_range(i, i + 1, S_[u], lnS_[u]);
if (state_[i] == ACTIVE)
book_[pos_[i]].price = price_[i];
else
activate(i);
changed_[n_changed_++] = i;
}
void learn_beta(int i, double sigma) {
const int u = under_[i];
if (!has_vol_[i] || searching_[i] || !has_S_[u]) {
beta_[i] = 0.0;
beta2_[i] = 0.0;
return;
}
const double dx = lnS_[u] - xv_[i];
const double dy = sigma - sigv_[i];
#if LLC_BETA_MODE == 3
beta_prev_ = shrunk_slope(bxx_[i], bxy_[i], byy_[i], bn_[i]);
#endif
constexpr double f = LLC_BETA_FORGET;
bxx_[i] = f * bxx_[i] + dx * dx;
bxy_[i] = f * bxy_[i] + dx * dy;
byy_[i] = f * byy_[i] + dy * dy;
bn_[i] = f * bn_[i] + 1.0;
constexpr double g = LLC_BETA_FORGET_GLOBAL;
gxx_ = g * gxx_ + dx * dx;
gxy_ = g * gxy_ + dx * dy;
gyy_ = g * gyy_ + dy * dy;
gn_ = g * gn_ + 1.0;
if (!bseen_[i]) bseen_[i] = 1, ++n_bseen_;
beta_g_ = shrunk_slope(gxx_, gxy_, gyy_, gn_);
#if LLC_BETA_MODE == 0
beta_[i] = shrunk_slope(bxx_[i], bxy_[i], byy_[i], bn_[i]);
#elif LLC_BETA_MODE == 1
const double dev = shrunk_slope(bxx_[i], bxy_[i] - beta_g_ * bxx_[i],
byy_[i] - 2.0 * beta_g_ * bxy_[i] + beta_g_ * beta_g_ * bxx_[i], bn_[i]);
beta_[i] = std::max(-LLC_BETA_MAX, std::min(LLC_BETA_MAX, beta_g_ + dev));
#elif LLC_BETA_MODE == 2
beta_[i] = beta_g_;
#else
if (g_mode0) { beta_[i] = shrunk_slope(bxx_[i], bxy_[i], byy_[i], bn_[i]); beta2_[i] = 0.0; } else
learn_ensemble(i, dx, dy);
#endif
}
struct Reg2 {
double s11 = 0, s12 = 0, s22 = 0, y1 = 0, y2 = 0, n = 0;
void add(double f, double r1, double r2, double y) {
s11 = f * s11 + r1 * r1;
s12 = f * s12 + r1 * r2;
s22 = f * s22 + r2 * r2;
y1 = f * y1 + r1 * y;
y2 = f * y2 + r2 * y;
n = f * n + 1.0;
}
void solve(double& b, double& c) const {
b = c = 0.0;
if (n < 10.0 || s11 <= 0.0) return;
const double det = s11 * s22 - s12 * s12;
if (det <= 1e-9 * s11 * s22 || s22 <= 0.0) {
b = y1 / s11;
return;
}
b = (y1 * s22 - y2 * s12) / det;
c = (y2 * s11 - y1 * s12) / det;
}
};
struct OptModel {
Reg2 smile;
double mse[4] = {0, 0, 0, 0};
double nerr = 0;
};
struct UndModel {
Reg2 smile;
double mse[4] = {0, 0, 0, 0};
double nerr = 0;
};
void learn_ensemble(int i, double dx, double dy) {
const int u = under_[i];
OptModel& om = om_[i];
UndModel& um = um_[u];
const double x_old = xv_[i] - lnK_[i];
const double x_new = lnS_[u] - lnK_[i];
const double r1 = dx, r2 = x_new * x_new - x_old * x_old;
const double b1 = beta_prev_;
double bu, cu, bo, co;
um.smile.solve(bu, cu);
om.smile.solve(bo, co);
const double pred[4] = {0.0, b1 * dx, bu * r1 + cu * r2, bo * r1 + co * r2};
constexpr double fe = LLC_ENS_FORGET;
for (int k = 0; k < 4; ++k) {
const double e = dy - pred[k];
om.mse[k] = fe * om.mse[k] + e * e;
um.mse[k] = fe * um.mse[k] + e * e;
}
om.nerr = fe * om.nerr + 1.0;
um.nerr = fe * um.nerr + 1.0;
om.smile.add(LLC_BETA_FORGET, r1, r2, dy);
um.smile.add(LLC_SMILE_FORGET_U, r1, r2, dy);
um.smile.solve(bu, cu);
om.smile.solve(bo, co);
const double b1n = shrunk_slope(bxx_[i], bxy_[i], byy_[i], bn_[i]);
const double* mse = om.nerr >= 10.0 ? om.mse : um.mse;
int best = 0;
for (int k = 1; k < 4; ++k)
if (mse[k] < mse[best] * LLC_ENS_MARGIN) best = k;
double k1 = 0.0, k2 = 0.0;
if (best == 1) k1 = b1n;
else if (best == 2) k1 = bu + 2.0 * cu * x_new, k2 = cu;
else if (best == 3) k1 = bo + 2.0 * co * x_new, k2 = co;
beta_[i] = std::max(-LLC_BETA_MAX, std::min(LLC_BETA_MAX, k1));
beta2_[i] = std::max(-LLC_BETA2_MAX, std::min(LLC_BETA2_MAX, k2));
}
static double shrunk_slope(double sxx, double sxy, double syy, double n) {
if (n <= 8.0 || sxx <= 0.0) return 0.0;
const double bh = sxy / sxx;
const double rss = std::max(0.0, syy - bh * sxy);
const double se2 = rss / std::max(1.0, n - 1.0) / sxx;
const double t2 = se2 > 0.0 ? bh * bh / se2 : 1e30;
const double shrink = t2 > 9.0 ? 1.0 - 9.0 / t2 : 0.0;
return std::max(-LLC_BETA_MAX, std::min(LLC_BETA_MAX, bh * shrink));
}
#if LLC_BETA_MODE == 2
void refresh_global_beta() {
for (int u = 0; u < nu_; ++u) {
bool any = false;
for (int i = u_start_[u]; i < u_start_[u + 1]; ++i)
if (has_vol_[i] && !searching_[i]) beta_[i] = beta_g_, any = true;
if (any && has_S_[u]) {
price_range(u_start_[u], u_start_[u + 1], S_[u], lnS_[u]);
for (int i = u_start_[u]; i < u_start_[u + 1]; ++i)
if (state_[i] == ACTIVE) book_[pos_[i]].price = price_[i];
}
}
}
#endif
inline void set_sigma(int i, double sigma) {
sigv_[i] = sigma > 1e-4 ? sigma : 1e-4;
beta_[i] = 0.0;
beta2_[i] = 0.0;
xv_[i] = lnS_[under_[i]];
}
void reprice_one(int i) {
const int u = under_[i];
price_range(i, i + 1, S_[u], lnS_[u]);
if (state_[i] == ACTIVE) book_[pos_[i]].price = price_[i];
}
inline int search_side() const { return (int)((calls_ / search_block_) & 1); }
void start_search() {
if (search_off_ || (double)n_vol_updates_ * LLC_SEARCH_VOL_SHARE_INV >= (double)calls_) return;
for (int u = 0; u < nu_; ++u) {
if (!has_S_[u]) continue;
double acc = 0;
int cnt = 0;
for (int i = u_start_[u]; i < u_start_[u + 1]; ++i)
if (has_vol_[i]) acc += std::log(sigv_[i]), ++cnt;
const double ls0 = cnt ? acc / cnt : std::log(LLC_SEARCH_SIGMA0);
for (int i = u_start_[u]; i < u_start_[u + 1]; ++i) {
if (has_vol_[i] || searching_[i] || state_[i] != NOT_READY) continue;
searching_[i] = 1;
ls_[i] = ls0;
h_[i] = LLC_SEARCH_H0;
dir_[i] = 0;
pairs_[i] = 0;
ssum_[2 * i] = ssum_[2 * i + 1] = 0;
scnt_[2 * i] = scnt_[2 * i + 1] = 0;
search_list_[n_search_++] = i;
set_sigma(i, std::exp(ls_[i] + (search_side() == 0 ? h_[i] : -h_[i])));
price_range(i, i + 1, S_[u], lnS_[u]);
activate(i);
}
}
}
void collect_search_scores(const std::vector<Score>& sc) {
const int m = (int)sc.size();
const bool aligned = (m == last_k_);
const int side = last_side_;
for (int c = 0; c < n_search_; ++c) {
const int i = search_list_[c];
if (!searching_[i] || state_[i] != ACTIVE) continue;
const int p = pos_[i];
if (p >= m) continue;
if (!aligned && sc[p].optionId != opt_id_[i]) continue;
ssum_[2 * i + side] += sc[p].profit;
search_win_ += sc[p].profit;
scnt_[2 * i + side] += 1;
}
}
void check_search_pays() {
search_bad_windows_ = search_win_ < 0.0 ? search_bad_windows_ + 1 : 0;
search_win_ = 0;
if (search_bad_windows_ < 3) return;
search_off_ = true;
for (int c = 0; c < n_search_; ++c) {
const int i = search_list_[c];
if (!searching_[i]) continue;
searching_[i] = 0;
if (state_[i] == ACTIVE) {
swap_book(pos_[i], nbook_ - 1);
--nbook_;
pos_[i] = -1;
}
state_[i] = SLEEPING;
}
n_search_ = 0;
}
void search_step() {
const int side = search_side();
int w = 0, n_dirty = 0;
for (int c = 0; c < n_search_; ++c) {
const int i = search_list_[c];
if (!searching_[i]) continue;
if (side == 0 && scnt_[2 * i] > 0 && scnt_[2 * i + 1] > 0) {
const double mp = ssum_[2 * i] / scnt_[2 * i];
const double mm = ssum_[2 * i + 1] / scnt_[2 * i + 1];
const double d = mp - mm;
double h = h_[i];
if (std::fabs(d) <= 1e-12 * (std::fabs(mp) + std::fabs(mm))) {
h = (mp < 0.0) ? h * 2.0 : h * 0.5;
} else {
const int dir = d > 0.0 ? 1 : -1;
ls_[i] += dir * h;
h = (dir == dir_[i]) ? h * 1.5 : h * 0.5;
dir_[i] = (int8_t)dir;
}
h_[i] = std::min(1.0, std::max((double)LLC_SEARCH_HMIN, h));
ls_[i] = std::min(std::log(4.0), std::max(std::log(0.005), ls_[i]));
ssum_[2 * i] = ssum_[2 * i + 1] = 0;
scnt_[2 * i] = scnt_[2 * i + 1] = 0;
if (++pairs_[i] >= LLC_SEARCH_GIVEUP && mp < 0.0 && mm < 0.0) {
searching_[i] = 0;
if (state_[i] == ACTIVE) {
swap_book(pos_[i], nbook_ - 1);
--nbook_;
pos_[i] = -1;
}
state_[i] = SLEEPING;
continue;
}
}
set_sigma(i, std::exp(ls_[i] + (side == 0 ? h_[i] : -h_[i])));
const int u = under_[i];
if (!u_dirty_[u]) u_dirty_[u] = 1, dirty_list_[n_dirty++] = u;
search_list_[w++] = i;
}
n_search_ = w;
for (int d = 0; d < n_dirty; ++d) {
const int u = dirty_list_[d];
u_dirty_[u] = 0;
const int lo = u_start_[u], hi = u_start_[u + 1];
price_range(lo, hi, S_[u], lnS_[u]);
for (int i = lo; i < hi; ++i)
if (state_[i] == ACTIVE) book_[pos_[i]].price = price_[i];
}
}
inline void activate(int i) {
if (!std::isfinite(price_[i])) return;
const int p = nbook_++;
book_[p] = Valuation{opt_id_[i], price_[i]};
book_idx_[p] = i;
pos_[i] = p;
state_[i] = ACTIVE;
}
inline void swap_book(int p, int q) {
if (p == q) return;
std::swap(book_[p], book_[q]);
std::swap(book_idx_[p], book_idx_[q]);
pos_[book_idx_[p]] = p;
pos_[book_idx_[q]] = q;
}
inline void put_to_sleep(int i) {
if (state_[i] != ACTIVE) return;
swap_book(pos_[i], nbook_ - 1);
--nbook_;
pos_[i] = -1;
state_[i] = SLEEPING;
sleepers_.push(Wake{calls_ + (uint64_t)backoff_[i], i, gen_[i]});
backoff_[i] = std::min<int>(backoff_[i] * 2, LLC_BACKOFF_MAX);
}
void promote_changed() {
int front = 0;
for (int c = 0; c < n_changed_ && front < klimit_; ++c) {
const int i = changed_[c];
if (state_[i] != ACTIVE) continue;
swap_book(front++, pos_[i]);
}
if ((calls_ & 4095) == 0) {
std::vector<int> order(book_idx_.begin() + front, book_idx_.begin() + nbook_);
std::sort(order.begin(), order.end(), [&](int x, int y) { return ev_[x] > ev_[y]; });
for (size_t j = 0; j < order.size(); ++j) swap_book(front + (int)j, pos_[order[j]]);
}
}
void read_scores(const std::vector<Score>& sc) {
const int m = (int)sc.size();
const Score* __restrict s = sc.data();
to_sleep_.clear();
if (m == last_k_) {
const int* __restrict bi = book_idx_.data();
if ((calls_ & 15) == 0 || klimit_ <= n_) {
for (int j = 0; j < m; ++j) ev_[bi[j]] = 0.5 * ev_[bi[j]] + 0.5 * s[j].profit;
}
for (int j0 = 0; j0 < m; j0 += 32) {
const int j1 = std::min(m, j0 + 32);
uint64_t acc = 0;
for (int j = j0; j < j1; ++j) {
uint64_t b;
std::memcpy(&b, &s[j].profit, 8);
acc |= b;
}
if (__builtin_expect(acc >> 63, 0)) {
for (int j = j0; j < j1; ++j)
if (s[j].profit < 0.0) on_loss(bi[j]);
}
}
} else {
for (int j = 0; j < m; ++j) {
const int i = idmap_opt_.find(s[j].optionId);
if (i < 0) continue;
ev_[i] = 0.5 * ev_[i] + 0.5 * s[j].profit;
if (s[j].profit < 0.0) on_loss(i);
}
}
for (int i : to_sleep_) put_to_sleep(i);
}
__attribute__((noinline)) void on_loss(int i) {
if (searching_[i]) return;
if (++neg_[i] >= LLC_NEG_K) to_sleep_.push_back(i);
}
void wake_sleepers() {
while (!sleepers_.empty() && sleepers_.top().at <= calls_) {
const Wake w = sleepers_.top();
sleepers_.pop();
if (w.gen != gen_[w.idx] || state_[w.idx] != SLEEPING) continue;
neg_[w.idx] = 0;
activate(w.idx);
}
}
void account(uint64_t cyc, int k) {
spent_ += (double)cyc + ext_call_cyc_ + ext_val_cyc_ * k;
win_cyc_ += (double)cyc + ext_call_cyc_ + ext_val_cyc_ * k;
win_k_ += k;
if ((calls_ & 255) != 0) return;
const double remaining = budget_cyc_ - spent_;
double calls_left = std::max((double)calls_, (double)LLC_N_GUESS - (double)calls_);
calls_left = std::min(calls_left, (double)LLC_N_MAX - (double)calls_);
calls_left = std::max(calls_left, 1000.0);
const double allowed = remaining / calls_left;
const double avg_cost = win_cyc_ / 256.0;
const double avg_k = win_k_ / 256.0;
win_cyc_ = 0;
win_k_ = 0;
if (avg_cost > allowed && search_block_ < 256) search_block_ *= 2;
else if (avg_cost < 0.5 * allowed && search_block_ > LLC_SEARCH_BLOCK) search_block_ /= 2;
if (avg_cost <= allowed) {
if (klimit_ <= n_) klimit_ = std::min(n_ + 1, (int)(klimit_ * 1.25) + 8);
return;
}
double target = avg_k * (allowed / avg_cost);
if (allowed <= 0) target = 0;
klimit_ = std::max(0, (int)std::min<double>(target, (double)n_ + 1));
}
struct Wake {
uint64_t at;
int idx;
uint32_t gen;
bool operator<(const Wake& o) const { return at > o.at; }
};
bool ready_ = false;
bool use_avx2_ = false;
bool use_avx512_ = false;
int n_ = 0, nu_ = 0;
std::vector<Option> reg_;
std::vector<int> opt_id_, under_, pos_, neg_, backoff_;
std::vector<uint32_t> gen_;
std::vector<double> T_, lnK_, sqrtT_, a_, sigv_, beta_, xv_, kdisc_, price_, ev_;
std::vector<double> bxx_, bxy_, byy_, bn_;
std::vector<uint8_t> bseen_;
std::vector<double> beta2_;
std::vector<OptModel> om_;
std::vector<UndModel> um_;
double beta_prev_ = 0;
int n_bseen_ = 0;
double gxx_ = 0, gxy_ = 0, gyy_ = 0, gn_ = 0, beta_g_ = 0;
std::vector<uint8_t> has_vol_, state_;
std::vector<uint8_t> searching_;
std::vector<int8_t> dir_;
std::vector<int> pairs_, scnt_, search_list_;
std::vector<double> ls_, h_, ssum_;
int n_search_ = 0;
uint64_t n_vol_updates_ = 0;
double search_win_ = 0;
int search_bad_windows_ = 0;
bool search_off_ = false;
uint64_t search_block_ = LLC_SEARCH_BLOCK;
std::vector<uint8_t> u_dirty_;
std::vector<int> dirty_list_;
int last_side_ = 0;
std::vector<int> u_start_;
std::vector<double> S_, lnS_;
std::vector<uint8_t> has_S_;
IdMap idmap_opt_, idmap_und_;
Kernel kernel_{};
std::vector<Valuation> book_;
std::vector<int> book_idx_;
int nbook_ = 0;
int last_k_ = 0;
std::vector<int> changed_;
int n_changed_ = 0;
std::vector<int> to_sleep_;
std::priority_queue<Wake> sleepers_;
uint64_t calls_ = 0;
double tsc_hz_ = 3e9;
double budget_cyc_ = 0, ext_call_cyc_ = 0, ext_val_cyc_ = 0;
double spent_ = 0, win_cyc_ = 0, win_k_ = 0;
int klimit_ = 1 << 30;
};
}
#undef LLC_BACKOFF0
#undef LLC_BACKOFF_MAX
#undef LLC_BETA
#undef LLC_BETA2_MAX
#undef LLC_BETA_FORGET
#undef LLC_BETA_FORGET_GLOBAL
#undef LLC_BETA_MAX
#undef LLC_BETA_MODE
#undef LLC_BUDGET_SEC
#undef LLC_ENS_FORGET
#undef LLC_ENS_MARGIN
#undef LLC_EXT_NS_CALL
#undef LLC_EXT_NS_VAL
#undef LLC_GATING
#undef LLC_NEG_K
#undef LLC_N_GUESS
#undef LLC_N_MAX
#undef LLC_REQUOTE_ALL
#undef LLC_SCORE_STRIDE
#undef LLC_SEARCH
#undef LLC_SEARCH_BLOCK
#undef LLC_SEARCH_GIVEUP
#undef LLC_SEARCH_H0
#undef LLC_SEARCH_HMIN
#undef LLC_SEARCH_SIGMA0
#undef LLC_SEARCH_START
#undef LLC_SEARCH_VOL_SHARE_INV
#undef LLC_SMILE_FORGET_U
#undef PROF
namespace vS {
#ifndef LLC_REQUOTE_ALL
#define LLC_REQUOTE_ALL 1
#endif
#ifndef LLC_STOPLOSS
#define LLC_STOPLOSS 1
#endif
#ifndef LLC_PROBE
#define LLC_PROBE 0
#endif
#ifndef LLC_SIBLING
#define LLC_SIBLING 0
#endif
#ifndef LLC_SEARCH
#define LLC_SEARCH 0
#endif
#ifndef LLC_SEARCH_AFTER
#define LLC_SEARCH_AFTER 1000
#endif
#ifndef LLC_UND_V
#define LLC_UND_V 1
#endif
#ifndef LLC_DECAY_FRAC
#define LLC_DECAY_FRAC 0.0
#endif
#ifndef LLC_BETA
#define LLC_BETA 1
#endif
#ifndef LLC_BETA_OPT
#define LLC_BETA_OPT 0
#endif
#ifndef LLC_CURV
#define LLC_CURV 0
#endif
#ifndef LLC_MODELSEL
#define LLC_MODELSEL 1
#endif
#ifndef LLC_MS_DECAY
#define LLC_MS_DECAY 0.99
#endif
#ifndef LLC_MS_FORGET
#define LLC_MS_FORGET 1.0
#endif
#ifndef LLC_ADDITIVE
#define LLC_ADDITIVE 1
#endif
#ifndef LLC_KALMAN
#define LLC_KALMAN 0
#endif
#ifndef LLC_DIAG
#define LLC_DIAG 0
#endif
#ifndef LLC_STEP0
#define LLC_STEP0 0.25
#endif
#ifndef LLC_RETRY_BASE
#define LLC_RETRY_BASE (1 << 20)
#endif
namespace {
constexpr double R = 0.04;
#if LLC_DIAG
static long diagSpun = 0;
static void diagSpin() {
if (diagSpun >= 800000) return;
++diagSpun;
auto t0 = std::chrono::steady_clock::now();
while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < 1.0) {}
}
[[maybe_unused]] static long diagCount = 0;
#endif
static inline double fexp_neg(double x) {
x = x < -700.0 ? -700.0 : x;
const double SHIFT = 6755399441055744.0;
double kd = x * 1.4426950408889634 + SHIFT;
uint64_t ki;
std::memcpy(&ki, &kd, 8);
kd -= SHIFT;
double r = x - kd * 0.6931471805599453;
r = r - kd * 2.3190468138462996e-17;
double p = 1.0 / 39916800.0;
p = p * r + 1.0 / 3628800.0;
p = p * r + 1.0 / 362880.0;
p = p * r + 1.0 / 40320.0;
p = p * r + 1.0 / 5040.0;
p = p * r + 1.0 / 720.0;
p = p * r + 1.0 / 120.0;
p = p * r + 1.0 / 24.0;
p = p * r + 1.0 / 6.0;
p = p * r + 0.5;
p = p * r + 1.0;
p = p * r + 1.0;
uint64_t bits = (ki + 1023ull) << 52;
double scale;
std::memcpy(&scale, &bits, 8);
return p * scale;
}
static inline double ncdf(double x) {
double a = std::fabs(x);
double e = fexp_neg(-0.5 * a * a);
double c;
if (a < 7.07106781186547) {
double n = 3.52624965998911e-02 * a + 0.700383064443688;
n = n * a + 6.37396220353165;
n = n * a + 33.912866078383;
n = n * a + 112.079291497871;
n = n * a + 221.213596169931;
n = n * a + 220.206867912376;
double d = 8.83883476483184e-02 * a + 1.75566716318264;
d = d * a + 16.064177579207;
d = d * a + 86.7807322029461;
d = d * a + 296.564248779674;
d = d * a + 637.333633378831;
d = d * a + 793.826512519948;
d = d * a + 440.413735824752;
c = e * n / d;
} else {
double b = a + 0.65;
b = a + 4.0 / b;
b = a + 3.0 / b;
b = a + 2.0 / b;
b = a + 1.0 / b;
c = e / (b * 2.506628274631);
}
return x > 0.0 ? 1.0 - c : c;
}
static inline double bsPrice(double S, double logS, double sig, double sqT, double logK,
double rT, double T, double dK) {
double ss = sig * sqT;
double d1 = (logS - logK + rT + 0.5 * sig * sig * T) / ss;
return S * ncdf(d1) - dK * ncdf(d1 - ss);
}
#define AVX2 __attribute__((target("avx2,fma"), always_inline)) static inline
AVX2 __m256d vexp_neg(__m256d x) {
x = _mm256_max_pd(x, _mm256_set1_pd(-700.0));
const __m256d SHIFT = _mm256_set1_pd(6755399441055744.0);
__m256d kd = _mm256_fmadd_pd(x, _mm256_set1_pd(1.4426950408889634), SHIFT);
__m256i ki = _mm256_castpd_si256(kd);
kd = _mm256_sub_pd(kd, SHIFT);
__m256d r = _mm256_fnmadd_pd(kd, _mm256_set1_pd(0.6931471805599453), x);
r = _mm256_fnmadd_pd(kd, _mm256_set1_pd(2.3190468138462996e-17), r);
__m256d p = _mm256_set1_pd(1.0 / 39916800.0);
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 3628800.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 362880.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 40320.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 5040.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 720.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 120.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 24.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 6.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.5));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
__m256i bits = _mm256_slli_epi64(_mm256_add_epi64(ki, _mm256_set1_epi64x(1023)), 52);
return _mm256_mul_pd(p, _mm256_castsi256_pd(bits));
}
AVX2 __m256d vtail(__m256d a, __m256d e) {
__m256d b = _mm256_add_pd(a, _mm256_set1_pd(0.65));
b = _mm256_add_pd(a, _mm256_div_pd(_mm256_set1_pd(4.0), b));
b = _mm256_add_pd(a, _mm256_div_pd(_mm256_set1_pd(3.0), b));
b = _mm256_add_pd(a, _mm256_div_pd(_mm256_set1_pd(2.0), b));
b = _mm256_add_pd(a, _mm256_div_pd(_mm256_set1_pd(1.0), b));
return _mm256_div_pd(e, _mm256_mul_pd(b, _mm256_set1_pd(2.506628274631)));
}
AVX2 void vncdf2(__m256d x1, __m256d x2, __m256d& out1, __m256d& out2) {
const __m256d sgnMask = _mm256_set1_pd(-0.0);
__m256d a1 = _mm256_andnot_pd(sgnMask, x1), a2 = _mm256_andnot_pd(sgnMask, x2);
__m256d e1 = vexp_neg(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(a1, a1)));
__m256d e2 = vexp_neg(_mm256_mul_pd(_mm256_set1_pd(-0.5), _mm256_mul_pd(a2, a2)));
#define F(acc,a,c) acc = _mm256_fmadd_pd(acc, a, _mm256_set1_pd(c))
#define NUM(n,a) __m256d n = _mm256_set1_pd(3.52624965998911e-02); \
F(n, a, 0.700383064443688); F(n, a, 6.37396220353165); F(n, a, 33.912866078383); \
F(n, a, 112.079291497871); F(n, a, 221.213596169931); F(n, a, 220.206867912376);
#define DEN(d,a) __m256d d = _mm256_set1_pd(8.83883476483184e-02); \
F(d, a, 1.75566716318264); F(d, a, 16.064177579207); F(d, a, 86.7807322029461); \
F(d, a, 296.564248779674); F(d, a, 637.333633378831); F(d, a, 793.826512519948); \
F(d, a, 440.413735824752);
NUM(n1, a1) NUM(n2, a2) DEN(d1, a1) DEN(d2, a2)
#undef NUM
#undef DEN
#undef F
__m256d inv = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(d1, d2));
__m256d c1 = _mm256_mul_pd(_mm256_mul_pd(e1, n1), _mm256_mul_pd(d2, inv));
__m256d c2 = _mm256_mul_pd(_mm256_mul_pd(e2, n2), _mm256_mul_pd(d1, inv));
const __m256d lim = _mm256_set1_pd(7.07106781186547);
__m256d far1 = _mm256_cmp_pd(a1, lim, _CMP_GE_OQ), far2 = _mm256_cmp_pd(a2, lim, _CMP_GE_OQ);
__m256d far = _mm256_or_pd(far1, far2);
if (!_mm256_testz_pd(far, far)) {
c1 = _mm256_blendv_pd(c1, vtail(a1, e1), far1);
c2 = _mm256_blendv_pd(c2, vtail(a2, e2), far2);
}
const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
out1 = _mm256_blendv_pd(c1, _mm256_sub_pd(one, c1), _mm256_cmp_pd(x1, zero, _CMP_GT_OQ));
out2 = _mm256_blendv_pd(c2, _mm256_sub_pd(one, c2), _mm256_cmp_pd(x2, zero, _CMP_GT_OQ));
}
__attribute__((target("avx2,fma"), noinline))
static void bsKernel(int lo, int hi, double S, double logS,
const double* __restrict sig, const double* __restrict sqT,
const double* __restrict logK, const double* __restrict rT,
const double* __restrict T, const double* __restrict dK,
double* __restrict out) {
int i = lo;
const __m256d vS = _mm256_set1_pd(S), vLogS = _mm256_set1_pd(logS), half = _mm256_set1_pd(0.5);
for (; i + 4 <= hi; i += 4) {
__m256d sg = _mm256_loadu_pd(sig + i);
__m256d ss = _mm256_mul_pd(sg, _mm256_loadu_pd(sqT + i));
__m256d num = _mm256_add_pd(_mm256_sub_pd(vLogS, _mm256_loadu_pd(logK + i)), _mm256_loadu_pd(rT + i));
num = _mm256_fmadd_pd(_mm256_mul_pd(half, _mm256_mul_pd(sg, sg)), _mm256_loadu_pd(T + i), num);
__m256d d1 = _mm256_div_pd(num, ss);
__m256d d2 = _mm256_sub_pd(d1, ss);
__m256d n1, n2;
vncdf2(d1, d2, n1, n2);
__m256d c = _mm256_fnmadd_pd(_mm256_loadu_pd(dK + i), n2, _mm256_mul_pd(vS, n1));
_mm256_storeu_pd(out + i, c);
}
for (; i < hi; ++i) out[i] = bsPrice(S, logS, sig[i], sqT[i], logK[i], rT[i], T[i], dK[i]);
}
struct IdMap {
int lo = 0;
std::vector<int> dense;
std::unordered_map<int, int> sparse;
bool useDense = false;
void build(const std::vector<std::pair<int, int>>& kv) {
if (kv.empty()) return;
int mn = kv[0].first, mx = kv[0].first;
for (auto& p : kv) { mn = std::min(mn, p.first); mx = std::max(mx, p.first); }
int64_t range = int64_t(mx) - mn + 1;
useDense = range <= (int64_t(1) << 24);
if (useDense) {
lo = mn;
dense.assign(size_t(range), -1);
for (auto& p : kv) dense[size_t(p.first - mn)] = p.second;
} else {
sparse.reserve(kv.size() * 2);
for (auto& p : kv) sparse[p.first] = p.second;
}
}
inline int get(int id) const {
if (useDense) {
int64_t k = int64_t(id) - lo;
return (k >= 0 && k < int64_t(dense.size())) ? dense[size_t(k)] : -1;
}
auto it = sparse.find(id);
return it == sparse.end() ? -1 : it->second;
}
};
constexpr int NK = 4;
constexpr double KGAIN[NK] = {0.0, 0.15, 0.4, 1.0};
}
class Solution : public ISolution {
std::vector<Option> reg;
bool built = false;
int n = 0;
std::vector<int> optId, under;
std::vector<double> K, T, sqT, logK, rT, dK;
std::vector<double> sigQ;
std::vector<double> price;
std::vector<double> sigBase;
std::vector<double> adj;
std::vector<double> step;
std::vector<double> sumP[2];
std::vector<int> cntP[2];
std::vector<int8_t> sgn, lastG, lossStreak;
std::vector<uint8_t> hasSig, benched;
std::vector<int> pos;
std::vector<int64_t> retryAt;
std::vector<int> backoff;
std::vector<double> bLev[NK];
std::vector<uint8_t> hasB;
int nu = 0;
std::vector<int> uLo, uHi;
std::vector<double> S, logS;
std::vector<uint8_t> hasS;
std::vector<double> lev[NK];
IdMap optMap, undMap;
std::vector<Valuation> book;
std::vector<int> bookIdx;
std::vector<int> losers;
int64_t callNo = 0;
double kErr[NK] = {0, 0, 0, 0};
int bestK = 0;
double innovAbs = 0.01;
double errAdj = 0, errNoAdj = 0;
int nEval = 0;
bool probeOn = LLC_PROBE;
long dgVknown = 0, dgVunk = 0, dgUunk = 0;
double sumV = 0; int nV = 0;
std::vector<int64_t> lastTick;
double rv2 = 0, rvGap = 0; long rvN = 0;
std::vector<double> lsV, lvLast;
std::vector<uint8_t> lsVok, lvOk;
double bSxx = 0, bSxy = 0, bSyy = 0; long bN = 0;
double beta = 0; bool betaOn = false;
double sRef = 0; bool sRefOk = false;
std::vector<double> oA11, oA12, oA22, oB1, oB2, bI, cI;
std::vector<int> oN;
static constexpr int NF = 4, NM = 5;
static constexpr unsigned MSET[NM] = {0x0, 0x1, 0x3, 0x5, 0xF};
std::vector<double> msG, msB, msTh, msSse;
std::vector<int> msBest;
std::vector<int64_t> tickV;
std::vector<double> lvSum;
std::vector<double> lvEst, dPrev;
std::vector<uint8_t> dPrevOk;
double kSdd = 0, kSdd1 = 0; long kNd = 0, kNd1 = 0;
double kGain = 1.0;
std::vector<uint8_t> srch, srchOff;
std::vector<double> lx, lh, pEma;
std::vector<int> nEvals;
std::vector<int64_t> offUntil;
std::vector<double> p0;
std::vector<uint8_t> needP0;
void buildState() {
built = true;
std::unordered_map<int, int> last;
for (int i = 0; i < int(reg.size()); ++i) last[reg[i].id] = i;
std::vector<Option> ops;
ops.reserve(last.size());
for (int i = 0; i < int(reg.size()); ++i)
if (last[reg[i].id] == i) ops.push_back(reg[i]);
std::stable_sort(ops.begin(), ops.end(),
[](const Option& a, const Option& b) { return a.underlyingId < b.underlyingId; });
n = int(ops.size());
auto rs = [&](auto& v, auto x) { v.assign(size_t(n) + 8, x); };
rs(optId, 0); rs(under, 0);
rs(K, 1.0); rs(T, 1.0); rs(sqT, 1.0); rs(logK, 0.0); rs(rT, 0.0); rs(dK, 1.0);
rs(sigQ, 0.2); rs(price, 0.0); rs(sigBase, 0.2); rs(adj, 0.0); rs(step, 0.0);
for (int s = 0; s < 2; ++s) { rs(sumP[s], 0.0); rs(cntP[s], 0); }
rs(sgn, int8_t(1)); rs(lastG, int8_t(0)); rs(lossStreak, int8_t(0)); rs(hasSig, uint8_t(0)); rs(benched, uint8_t(0));
rs(pos, -1); rs(retryAt, int64_t(0)); rs(backoff, LLC_RETRY_BASE); rs(hasB, uint8_t(0));
rs(srch, uint8_t(0)); rs(srchOff, uint8_t(0)); rs(lx, 0.0); rs(lh, 0.0); rs(pEma, 0.0);
rs(nEvals, 0);
rs(oA11, 0.0); rs(oA12, 0.0); rs(oA22, 0.0); rs(oB1, 0.0); rs(oB2, 0.0); rs(bI, 0.0); rs(cI, 0.0); rs(oN, 0);
msG.assign(size_t(n) * 16 + 16, 0.0); msB.assign(size_t(n) * 4 + 4, 0.0);
msTh.assign(size_t(n) * NM * 4 + 32, 0.0); msSse.assign(size_t(n) * NM + 8, 0.0);
rs(msBest, 0); rs(tickV, int64_t(0)); rs(lvSum, 0.0);
rs(lvEst, 0.0); rs(dPrev, 0.0); rs(dPrevOk, uint8_t(0));
rs(lsV, 0.0); rs(lvLast, 0.0); rs(lsVok, uint8_t(0)); rs(lvOk, uint8_t(0)); rs(p0, 0.0); rs(needP0, uint8_t(0)); rs(offUntil, int64_t(0));
for (int k = 0; k < NK; ++k) rs(bLev[k], 0.0);
std::vector<std::pair<int, int>> okv, ukv;
for (int i = 0; i < n; ++i) {
const Option& o = ops[size_t(i)];
optId[i] = o.id;
double t = o.tte > 1e-9 ? o.tte : 1e-9;
K[i] = double(o.strike) > 1e-9 ? double(o.strike) : 1e-9;
T[i] = t; sqT[i] = std::sqrt(t); logK[i] = std::log(K[i]);
rT[i] = R * t; dK[i] = K[i] * std::exp(-R * t);
okv.push_back({o.id, i});
if (i == 0 || o.underlyingId != ops[size_t(i - 1)].underlyingId) {
ukv.push_back({o.underlyingId, nu});
uLo.push_back(i); uHi.push_back(i);
++nu;
}
under[i] = nu - 1;
uHi.back() = i + 1;
}
optMap.build(okv);
undMap.build(ukv);
lastTick.assign(size_t(nu), 0);
S.assign(size_t(nu), 0.0); logS.assign(size_t(nu), 0.0); hasS.assign(size_t(nu), 0);
for (int k = 0; k < NK; ++k) lev[k].assign(size_t(nu), 0.0);
book.reserve(size_t(n) + 8);
bookIdx.reserve(size_t(n) + 8);
losers.reserve(size_t(n) + 8);
reg.clear(); reg.shrink_to_fit();
}
inline void quote(int i) {
if (pos[i] < 0) {
pos[i] = int(book.size());
book.push_back({optId[i], price[i]});
bookIdx.push_back(i);
}
else book[size_t(pos[i])].price = price[i];
}
inline void unquote(int i) {
int p = pos[i];
if (p < 0) return;
int lastIdx = bookIdx.back();
book[size_t(p)] = book.back();
bookIdx[size_t(p)] = lastIdx;
pos[lastIdx] = p;
book.pop_back();
bookIdx.pop_back();
pos[i] = -1;
}
inline void msFeatures(int i, double s1, double s0, double* f) const {
double a = s1 - sRef, b = s0 - sRef;
f[0] = s1 - s0;
f[1] = a * a - b * b;
f[2] = double(callNo - tickV[i]) * 1e-3;
f[3] = f[2] * (lvLast[i] - lvSum[i] / std::max(1, oN[i] + 1));
}
void msSolve(int i, int m) {
unsigned set = MSET[m];
int idx[NF], k = 0;
for (int j = 0; j < NF; ++j) if (set >> j & 1u) idx[k++] = j;
double* th = &msTh[(size_t(i) * NM + m) * 4];
for (int j = 0; j < 4; ++j) th[j] = 0.0;
if (k == 0) return;
const double* G = &msG[size_t(i) * 16];
const double* bv = &msB[size_t(i) * 4];
double A[4][5];
for (int r = 0; r < k; ++r) {
for (int c = 0; c < k; ++c) A[r][c] = G[idx[r] * 4 + idx[c]];
A[r][r] += 1e-6 * G[idx[r] * 4 + idx[r]] + 1e-30;
A[r][k] = bv[idx[r]];
}
for (int c = 0; c < k; ++c) {
int pv = c;
for (int r = c + 1; r < k; ++r) if (std::fabs(A[r][c]) > std::fabs(A[pv][c])) pv = r;
if (!(std::fabs(A[pv][c]) > 1e-300)) return;
if (pv != c) for (int q = 0; q <= k; ++q) std::swap(A[c][q], A[pv][q]);
for (int r = 0; r < k; ++r) {
if (r == c) continue;
double fct = A[r][c] / A[c][c];
for (int q = c; q <= k; ++q) A[r][q] -= fct * A[c][q];
}
}
for (int r = 0; r < k; ++r) {
double v = A[r][k] / A[r][r];
th[idx[r]] = std::isfinite(v) ? v : 0.0;
}
}
void msUpdate(int i, double s1, double s0, double y) {
double f[NF];
msFeatures(i, s1, s0, f);
double* sse = &msSse[size_t(i) * NM];
for (int m = 0; m < NM; ++m) {
const double* th = &msTh[(size_t(i) * NM + m) * 4];
double pred = 0;
for (int j = 0; j < NF; ++j) pred += th[j] * f[j];
pred = std::clamp(pred, -3.0, 3.0);
double e = y - pred;
sse[m] = LLC_MS_DECAY * sse[m] + e * e;
}
double* G = &msG[size_t(i) * 16];
double* bv = &msB[size_t(i) * 4];
for (int r = 0; r < NF; ++r) {
for (int c = 0; c < NF; ++c) G[r * 4 + c] = LLC_MS_FORGET * G[r * 4 + c] + f[r] * f[c];
bv[r] = LLC_MS_FORGET * bv[r] + f[r] * y;
}
++oN[i];
for (int m = 1; m < NM; ++m) msSolve(i, m);
int best = 0;
if (oN[i] >= 15)
for (int m = 1; m < NM; ++m) if (sse[m] < sse[best] * (best == 0 ? 0.95 : 1.0)) best = m;
msBest[i] = best;
}
inline double modelDelta(int i, double s1, double s0) const {
if (LLC_MODELSEL) {
int m = msBest[i];
if (m == 0) return 0.0;
double f[NF];
msFeatures(i, s1, s0, f);
const double* th = &msTh[(size_t(i) * NM + m) * 4];
double d = 0;
for (int j = 0; j < NF; ++j) d += th[j] * f[j];
return std::clamp(d, -3.0, 3.0);
}
if (LLC_BETA_OPT && oN[i] >= 20) {
double d = bI[i] * (s1 - s0);
if (LLC_CURV) { double a = s1 - sRef, b = s0 - sRef; d += cI[i] * (a * a - b * b); }
return d;
}
return betaOn ? beta * (s1 - s0) : 0.0;
}
void fitOption(int i) {
double n = std::max(1, oN[i]);
double l1 = 10.0 * oA11[i] / n + 1e-18, l2 = 10.0 * oA22[i] / n + 1e-18;
double bg = betaOn ? beta : 0.0;
if (!LLC_CURV) { bI[i] = std::clamp((oB1[i] + l1 * bg) / (oA11[i] + l1), -20.0, 20.0); cI[i] = 0; return; }
double a11 = oA11[i] + l1, a12 = oA12[i], a22 = oA22[i] + l2;
double r1 = oB1[i] + l1 * bg, r2 = oB2[i];
double det = a11 * a22 - a12 * a12;
if (!(std::fabs(det) > 1e-300)) { bI[i] = bg; cI[i] = 0; return; }
bI[i] = std::clamp((r1 * a22 - r2 * a12) / det, -50.0, 50.0);
cI[i] = std::clamp((a11 * r2 - a12 * r1) / det, -500.0, 500.0);
}
inline void setSigQ(int i) {
double d = probeOn ? step[i] : 0.0;
double s = sigBase[i] + (probeOn ? adj[i] : 0.0) + sgn[i] * d;
if (LLC_BETA && !srch[i]) {
int u = under[i];
if (!lsVok[i] && hasS[size_t(u)]) { lsV[i] = logS[size_t(u)]; lsVok[i] = 1; }
if (lsVok[i]) {
double md = modelDelta(i, logS[size_t(u)], lsV[i]);
if (LLC_ADDITIVE) s += md; else s *= std::exp(md);
}
}
sigQ[i] = s < 1e-4 ? 1e-4 : s;
}
inline bool canQuote(int i) const { return hasSig[i] && hasS[under[i]] && std::isfinite(price[i]); }
void probeDecide(int i) {
if (!probeOn || cntP[0][i] == 0 || cntP[1][i] == 0) return;
double mp = sumP[0][i] / cntP[0][i], mm = sumP[1][i] / cntP[1][i];
double tol = 1e-9 * (std::fabs(mp) + std::fabs(mm));
int g = mp > mm + tol ? 1 : (mm > mp + tol ? -1 : 0);
double st = step[i];
if (g == 0) st *= 0.5;
else {
st *= (g == lastG[i]) ? 1.3 : 0.6;
adj[i] += g * step[i];
double cap = 0.5 * sigBase[i];
adj[i] = std::clamp(adj[i], -cap, cap);
}
lastG[i] = int8_t(g);
double smax = std::max(0.02, 0.1 * sigBase[i]);
step[i] = std::clamp(st, 2e-5, smax);
sumP[0][i] = sumP[1][i] = 0; cntP[0][i] = cntP[1][i] = 0;
}
void startSearch() {
double guess = nV > 0 ? sumV / nV : 0.3;
for (int i = 0; i < n; ++i) {
if (hasSig[i] || !hasS[under[i]]) continue;
srch[i] = 1; hasSig[i] = 1;
lx[i] = std::log(guess); lh[i] = 0.4; lastG[i] = 0;
sumP[0][i] = sumP[1][i] = 0; cntP[0][i] = cntP[1][i] = 0;
sigBase[i] = guess;
}
}
void searchDecide(int i) {
if (cntP[0][i] == 0 || cntP[1][i] == 0) return;
double mp = sumP[0][i] / cntP[0][i], mm = sumP[1][i] / cntP[1][i];
double tol = 1e-9 * (std::fabs(mp) + std::fabs(mm));
int g = mp > mm + tol ? 1 : (mm > mp + tol ? -1 : 0);
double h = lh[i];
if (g == 0) h *= (mp < 0.0 ? 2.0 : 0.5);
else {
lx[i] += g * h;
h *= (g == lastG[i]) ? 1.5 : 0.5;
}
lastG[i] = int8_t(g);
lh[i] = std::clamp(h, 0.001, 1.0);
lx[i] = std::clamp(lx[i], std::log(0.01), std::log(5.0));
sigBase[i] = std::exp(lx[i]);
sumP[0][i] = sumP[1][i] = 0; cntP[0][i] = cntP[1][i] = 0;
}
void repriceUnderlying(int u) {
int lo = uLo[size_t(u)], hi = uHi[size_t(u)];
for (int i = lo; i < hi; ++i) {
if (!hasSig[i]) continue;
if (srch[i]) {
searchDecide(i);
sgn[i] = int8_t(-sgn[i]);
sigQ[i] = std::exp(lx[i] + sgn[i] * lh[i]);
continue;
}
probeDecide(i);
sgn[i] = int8_t(-sgn[i]);
setSigQ(i);
}
bsKernel(lo, hi, S[size_t(u)], logS[size_t(u)], sigQ.data(), sqT.data(), logK.data(),
rT.data(), T.data(), dK.data(), price.data());
for (int i = lo; i < hi; ++i) {
if (!hasSig[i] || !std::isfinite(price[i])) continue;
if (srch[i]) {
if (srchOff[i]) {
if (callNo < offUntil[i]) continue;
srchOff[i] = 0; pEma[i] = 0; nEvals[i] = 0; lh[i] = std::max(lh[i], 0.05);
}
quote(i);
continue;
}
if (benched[i]) {
if (callNo < retryAt[i]) continue;
benched[i] = 0;
}
quote(i);
}
}
void onVol(int i, double v) {
if (!(v > 0.0) || !std::isfinite(v)) return;
int u = under[i];
double lv = std::log(v);
if (hasSig[i]) {
double predNo = sigBase[i];
double predAdj = sigBase[i] + adj[i];
double eNo = std::fabs(v - predNo), eAdj = std::fabs(v - predAdj);
errNoAdj += eNo; errAdj += eAdj; ++nEval;
if (LLC_PROBE && nEval >= 40 && probeOn && errAdj > 1.15 * errNoAdj) probeOn = false;
innovAbs = 0.95 * innovAbs + 0.05 * eNo;
}
bool siblingsMoved = false;
if (LLC_SIBLING) {
for (int k = 0; k < NK; ++k) {
double& L = lev[k][size_t(u)];
if (hasB[i]) {
double innov = lv - (bLev[k][i] + L);
kErr[k] = 0.98 * kErr[k] + 0.02 * innov * innov;
L += KGAIN[k] * innov;
}
bLev[k][i] = lv - L;
}
hasB[i] = 1;
int bk = 0;
for (int k = 1; k < NK; ++k) if (kErr[k] < kErr[bk]) bk = k;
bestK = bk;
siblingsMoved = KGAIN[bestK] > 0.0;
}
sumV += v; ++nV;
if (hasS[size_t(u)] && !sRefOk) { sRef = logS[size_t(u)]; sRefOk = true; }
double newEst = lv;
if (lvOk[i] && lsVok[i] && hasS[size_t(u)]) {
double s1 = logS[size_t(u)], s0 = lsV[i];
double md = LLC_BETA ? modelDelta(i, s1, s0) : 0.0;
if (LLC_KALMAN) {
double d = (lv - lvLast[i]) - md;
kSdd += d * d; ++kNd;
if (dPrevOk[i]) { kSdd1 += d * dPrev[i]; ++kNd1; }
dPrev[i] = d; dPrevOk[i] = 1;
if (kNd1 >= 100) {
double var = kSdd / kNd, cov = kSdd1 / kNd1;
double r = std::max(0.0, -cov), q = std::max(var - 2.0 * r, 1e-14);
if (r > 0.05 * var) {
double P = 0.5 * (q + std::sqrt(q * q + 4.0 * q * r));
kGain = P / (P + r);
} else kGain = 1.0;
}
double pred = lvEst[i] + md;
newEst = pred + kGain * (lv - pred);
}
if (LLC_MODELSEL) msUpdate(i, s1, s0, LLC_ADDITIVE ? v - std::exp(lvLast[i]) : lv - lvLast[i]);
else if (LLC_BETA_OPT) {
double x1 = s1 - s0, y1 = lv - lvLast[i];
double a = s1 - sRef, b = s0 - sRef, x2 = a * a - b * b;
oA11[i] += x1 * x1; oA12[i] += x1 * x2; oA22[i] += x2 * x2;
oB1[i] += x1 * y1; oB2[i] += x2 * y1; ++oN[i];
}
}
if (lvOk[i] && lsVok[i] && hasS[size_t(u)]) {
double y = lv - lvLast[i], x = logS[size_t(u)] - lsV[i];
#if LLC_DIAG == 5
if (std::fabs(y - beta * x) < std::fabs(y)) diagSpin();
#endif
bSxx += x * x; bSxy += x * y; bSyy += y * y; ++bN;
if (bN >= 50 && bSxx > 0 && bSyy > 0) {
double c = bSxy / std::sqrt(bSxx * bSyy);
double t = c * std::sqrt((bN - 2) / std::max(1e-12, 1.0 - c * c));
beta = std::clamp(bSxy / bSxx, -10.0, 10.0);
betaOn = std::fabs(t) > 3.0;
}
}
if (LLC_BETA_OPT && !LLC_MODELSEL) fitOption(i);
tickV[i] = callNo; lvSum[i] += lv;
lvLast[i] = lv; lvOk[i] = 1;
lvEst[i] = newEst;
if (hasS[size_t(u)]) { lsV[i] = logS[size_t(u)]; lsVok[i] = 1; } else lsVok[i] = 0;
srch[i] = 0; srchOff[i] = 0;
needP0[i] = 1;
sigBase[i] = std::exp(newEst);
hasSig[i] = 1;
adj[i] = 0.0;
step[i] = std::clamp(LLC_STEP0 * innovAbs, 1e-4, 0.02);
lastG[i] = 0;
sumP[0][i] = sumP[1][i] = 0; cntP[0][i] = cntP[1][i] = 0;
benched[i] = 0;
backoff[i] = LLC_RETRY_BASE;
if (!hasS[size_t(u)]) return;
if (siblingsMoved) {
int lo = uLo[size_t(u)], hi = uHi[size_t(u)];
const std::vector<double>& b = bLev[bestK];
double L = lev[bestK][size_t(u)];
for (int j = lo; j < hi; ++j)
if (j != i && hasSig[j] && hasB[j]) { sigBase[j] = std::exp(b[j] + L); setSigQ(j); }
sigBase[i] = v;
setSigQ(i);
bsKernel(lo, hi, S[size_t(u)], logS[size_t(u)], sigQ.data(), sqT.data(), logK.data(),
rT.data(), T.data(), dK.data(), price.data());
for (int j = lo; j < hi; ++j)
if (hasSig[j] && !benched[j] && std::isfinite(price[j])) quote(j);
if (std::isfinite(price[i])) quote(i);
} else {
setSigQ(i);
price[i] = bsPrice(S[size_t(u)], logS[size_t(u)], sigQ[i], sqT[i], logK[i], rT[i], T[i], dK[i]);
if (std::isfinite(price[i])) quote(i);
}
}
void onScores(const std::vector<Score>& sc) {
const size_t nb = bookIdx.size();
losers.clear();
for (size_t k = 0; k < sc.size(); ++k) {
const Score& s = sc[k];
int i = (k < nb && optId[bookIdx[k]] == s.optionId) ? bookIdx[k] : optMap.get(s.optionId);
if (i < 0) continue;
double p = s.profit;
#if LLC_DIAG == 1
if (p < 0.0) diagSpin();
#elif LLC_DIAG == 2
if ((++diagCount & 31) == 0) diagSpin();
#elif LLC_DIAG == 3
if (srch[i] && p > 0.0) diagSpin();
#endif
int side = sgn[i] > 0 ? 0 : 1;
sumP[side][i] += p; ++cntP[side][i];
if (srch[i]) {
++nEvals[i];
pEma[i] = nEvals[i] < 1000 ? pEma[i] + (p - pEma[i]) / nEvals[i] : pEma[i] + 0.001 * (p - pEma[i]);
if (nEvals[i] >= 3000 && pEma[i] < 0.0) losers.push_back(i);
continue;
}
if (p < 0.0) {
#if LLC_DIAG == 12
diagSpin();
#elif LLC_DIAG == 13
if (needP0[i]) diagSpin();
#endif
if (LLC_STOPLOSS) losers.push_back(i);
}
else if (needP0[i]) { p0[i] = p; needP0[i] = 0; }
else if (LLC_DECAY_FRAC > 0.0 && p < LLC_DECAY_FRAC * p0[i]) losers.push_back(i);
else lossStreak[i] = 0;
}
for (int i : losers) {
if (pos[i] < 0) continue;
if (srch[i]) {
unquote(i);
srchOff[i] = 1;
offUntil[i] = callNo + 50000;
continue;
}
int s = sgn[i], other = s > 0 ? 1 : 0;
if (probeOn && cntP[other][i] > 0 && sumP[other][i] > 0.0 && lossStreak[i] == 0) {
adj[i] -= 2.0 * s * step[i];
double cap = 0.5 * sigBase[i];
adj[i] = std::clamp(adj[i], -cap, cap);
sumP[0][i] = sumP[1][i] = 0; cntP[0][i] = cntP[1][i] = 0;
lastG[i] = 0;
setSigQ(i);
int u = under[i];
price[i] = bsPrice(S[size_t(u)], logS[size_t(u)], sigQ[i], sqT[i], logK[i], rT[i], T[i], dK[i]);
lossStreak[i] = 1;
if (std::isfinite(price[i])) { quote(i); continue; }
}
lossStreak[i] = 0;
unquote(i);
benched[i] = 1;
retryAt[i] = callNo + backoff[i];
backoff[i] = std::min(backoff[i] * 2, 1 << 20);
}
}
public:
Solution() {
#ifdef M_MMAP_THRESHOLD
mallopt(M_MMAP_THRESHOLD, 256 * 1024 * 1024);
mallopt(M_TRIM_THRESHOLD, 512 * 1024 * 1024);
mallopt(M_TOP_PAD, 64 * 1024 * 1024);
#endif
reg.reserve(4096);
}
void registerOption(const Option& option) override { reg.push_back(option); }
std::vector<Valuation> process(const ParameterUpdate& update,
const std::vector<Score>& previousScores) override {
if (!built) buildState();
++callNo;
if (!previousScores.empty()) onScores(previousScores);
if (LLC_SEARCH && callNo % LLC_SEARCH_AFTER == 0) startSearch();
#if LLC_DIAG == 14 || LLC_DIAG == 15
if (callNo == 1000000) {
double ms = LLC_DIAG == 14
? 4.0 * std::min(n, 99) + (dgVknown > 0 ? 400.0 : 0.0)
: 0.01 * double(std::min(dgVunk, 30000L)) + (dgUunk > 0 ? 400.0 : 0.0);
auto t0 = std::chrono::steady_clock::now();
while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms) {}
}
#endif
#if LLC_DIAG == 8 || LLC_DIAG == 9
if (callNo == 2000) {
double sv = LLC_DIAG == 8 ? std::sqrt(rv2 / std::max(1.0, rvGap))
: std::sqrt(rv2 / std::max(1L, rvN));
double x = nV > 0 ? sv / (sumV / nV) : sv;
double ms = std::clamp(50.0 * (std::log10(std::max(x, 1e-12)) + 8.0), 0.0, 500.0);
auto t0 = std::chrono::steady_clock::now();
while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms) {}
}
#endif
if (update.type == 'U') {
int u = undMap.get(update.instrumentId);
double v = update.value;
if (u < 0) ++dgUunk;
if (u >= 0 && v > 0.0 && std::isfinite(v)) {
double l = std::log(v);
if (hasS[size_t(u)]) {
double r = l - logS[size_t(u)];
rv2 += r * r; rvGap += double(callNo - lastTick[size_t(u)]); ++rvN;
}
lastTick[size_t(u)] = callNo;
S[size_t(u)] = v; logS[size_t(u)] = l; hasS[size_t(u)] = 1;
repriceUnderlying(u);
}
} else if (update.type == 'V') {
int i = optMap.get(update.instrumentId);
if (i >= 0) { ++dgVknown; onVol(i, update.value); }
else if (++dgVunk, LLC_UND_V) {
int u = undMap.get(update.instrumentId);
if (u >= 0) {
#if LLC_DIAG == 4
for (int k = 0; k < 20; ++k) diagSpin();
#endif
for (int j = uLo[size_t(u)]; j < uHi[size_t(u)]; ++j) onVol(j, update.value);
}
}
}
if (LLC_REQUOTE_ALL) return book;
std::vector<Valuation> out;
if (update.type == 'U') {
int u = undMap.get(update.instrumentId);
if (u >= 0)
for (int i = uLo[size_t(u)]; i < uHi[size_t(u)]; ++i)
if (pos[i] >= 0) out.push_back(book[size_t(pos[i])]);
} else {
int i = optMap.get(update.instrumentId);
if (i >= 0 && pos[i] >= 0) out.push_back(book[size_t(pos[i])]);
}
return out;
}
};
}
class Solution : public ISolution {
hS::Solution* h_; vS::Solution* v_;
bool useV_ = false, decided_ = false;
long calls_ = 0, nv_ = 0;
std::vector<Score> empty_;
public:
Solution() { h_ = new hS::Solution(); v_ = new vS::Solution(); }
long nopt_ = 0;
void registerOption(const Option& o) override { ++nopt_; h_->registerOption(o); v_->registerOption(o); }
std::vector<Valuation> process(const ParameterUpdate& u, const std::vector<Score>& ps) override {
if (useV_) return v_->process(u, ps);
if (decided_) return h_->process(u, ps);
++calls_;
if (u.type == 'V') ++nv_;
std::vector<Valuation> out = h_->process(u, ps);
v_->process(u, empty_);
if (calls_ == 300) {
decided_ = true;
if (nv_ > 0 && nv_ < 30) useV_ = true;
if (nv_ >= 120 && nv_ <= 195 && nopt_ >= 3) hS::g_mode0 = true;   // test 2: ~52% vol updates, ~3 options
}
return out;
}
};
ISolution* createSolution() { return new Solution(); }
