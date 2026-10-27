# Low Latency Challenge

| Rule | |
|---|---|
| Submission | One `.cpp` file with your `Solution` class and `createSolution()` (no `main`), at most 64 KB |
| Compiler | `g++-13 -std=c++20 -O3 -static`, Linux x86-64 |
| Threads | Single-threaded only. No threads, no child processes, no file or network access |
| Memory | 2 GB per test |
| Time budget | 1 second per test, counting only the time spent inside `process()` |
| Setup | `createSolution()` and all `registerOption()` calls must finish within 2 seconds (not counted in the budget) |
| Test cases | 6, in increasing order of difficulty. Your total is the sum of the test scores |
| Rate limit | One submission per minute per team |
| Leaderboard | Your best submission counts. Ties go to the team that submitted that score first |

## Introduction

Welcome to the Low Latency Challenge! In this challenge, you have to design a high-performance trading system. Speed and accuracy both matter, and there is no perfect solution. The higher your score is, the better!

## Black–Scholes formula

A **call option** is a financial contract that gives its owner the right to buy an asset (e.g. a stock) at a fixed price on a future date. The **Black–Scholes formula** is a mathematical model that estimates the *price* of an option, given several inputs.

The price of the call option is:

$$
C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)
$$

where:

$$
d_1=\frac{\ln(S/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\quad
d_2=d_1-\sigma\sqrt{T}
$$

- $S$: underlying asset price
- $K$: strike price
- $T$: time to maturity (in years)
- $r$: risk-free interest rate
- $\sigma$: volatility
- $N(\cdot)$: standard normal cumulative distribution function

(You do not need to derive or deeply understand these formulas for the challenge.)

## Problem statement

You are designing an execution system at a trading firm. The execution system knows a list of options that we want to trade, and tries to calculate accurate prices for as many options as possible. For each option, given the inputs of the Black–Scholes formula, it is possible to calculate the price of the option. This execution system is responsible for *collecting* inputs from other systems, and finally sending a price to the exchange for trading.

For the execution system, here are the inputs:

- $S$: underlying asset price. This updates frequently, as a `ParameterUpdate` input.
- $K$: strike price. This is constant for an option, and is provided when registering the option on startup, as an `Option` input.
- $T$: time to expiry in years (TTE). This is constant for an option, and is provided when registering the option on startup, as an `Option` input.
- $r$: risk-free interest rate. This is a constant for *all* options, with the value `0.04`.
- $\sigma$: volatility. This updates infrequently, as a `ParameterUpdate` input.
- $N(\cdot)$: the standard normal [cumulative distribution function](https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function).

Each time you receive a `ParameterUpdate`, you may return a list of `Valuation`s, which represent your best guess of the price of options. You may price any subset of the options. It may not always be in your best interest to price an option: it's easy to lose money in the stock market if your valuation is wrong!

### About volatility

In practice, asset price updates are easy to calculate, while volatility updates are harder. Even though both parameters are constantly changing, our execution system may not always know the latest volatility.

For example, on one option, you could receive updates in this order: U1, V1, U2, U3, U4, V2. This means you received an underlying update, then a volatility update, then 3 underlying updates before the next volatility update. Between the two volatility updates the volatility could still change in reality; our trading system is just not fast enough to calculate it.

## Scoring

When you send a price for an option, a score is calculated. The exact scoring function is not revealed, but here are some characteristics:

- The grader knows a reference price for each option. The closer you are to the reference price, the higher your score, and the farther you are, the lower your score.
- If your price is farther than some threshold from the reference price, you start losing money. At that point it would have been better not to send a valuation at all!
- Some options have more people trading, so the amount of money that can be gained (or lost!) in them is larger than in others.
- Not sending a valuation scores nothing, so a test score can be negative if you send bad prices.

How valuations are counted:

- Each option is scored **at most once per `process` call**. If you return several valuations for the same `optionId` in one call, only the first valid one is scored and the rest are ignored.
- Valuations with an unknown `optionId`, or a price that is NaN or infinite, are ignored. They don't use up that option's one valuation for the call.
- At most (number of options + 64) valuations per call are read. Anything beyond that is ignored.

### Feedback: `previousScores`

Each `process` call passes you `previousScores`: one `Score` for every valuation that was scored on the **previous** call, in the order you returned them. `profit` is what that valuation earned (positive) or lost (negative). Ignored valuations don't appear.

`previousScores` is empty on the first call, and on any call after one where you returned nothing. Scores are only ever reported once, for the call right before. You may use these scores as you wish, or ignore them completely.

### Test and total score

Your score for a test is computed from the profit of all your valuations across all of its inputs. Your total score is the sum of your 6 test scores, and your best total over all your submissions is what counts for the leaderboard.

## Implementation

You need to implement a program that conforms to this API (included in `api.hpp`):

```cpp
struct ParameterUpdate {
    char type;
    double value;
    int instrumentId;
};

struct Valuation {
    int optionId;
    double price;
};

struct Option {
    int id;
    int underlyingId;
    int strike;
    double tte;
};

struct Score {
    int optionId;
    double profit;
};

class ISolution {
public:
    virtual ~ISolution() = default;

    virtual void registerOption(const Option& option) = 0;
    virtual std::vector<Valuation> process(const ParameterUpdate& update, const std::vector<Score>& previousScores) = 0;
};

ISolution* createSolution();
```

`ParameterUpdate`:

- `type` is `U` or `V`. `U` is an underlying price update, and `V` is a volatility update.
- `value` is the new value.
- `instrumentId` is the `optionId` for a volatility update, and the `underlyingId` for an underlying price update.

The grader interacts with your program as follows, separately for each test:

1. It starts a fresh copy of your program and calls `createSolution()` once.
2. It calls `registerOption` once for every option in the test case. Several options can share the same underlying.
3. It then calls `process` once for every `ParameterUpdate`, in order. Each call also gives you the scores of the `Valuation`s you returned on the previous call.

Nothing carries over between tests: each test runs in its own process.

## Starter kit

Download the starter kit with the button at the top of this page. It contains `api.hpp`, `solution_template.cpp` and `test_main.cpp`. Build on top of `solution_template.cpp` by implementing the two methods of the `Solution` class. `test_main.cpp` is a small local runner with a fake scoring function (every valuation scores 42). You can change it to create your own test cases, but you only submit the `.cpp` file with the `Solution` class.

To test locally:

```sh
g++ -std=c++20 -O3 solution_template.cpp test_main.cpp -o solution
./solution
```

## Execution environment

### Compilation

Your file is compiled on Linux x86-64 (Ubuntu 24.04) with:

```sh
g++-13 -std=c++20 -O3 -static solution.cpp <grader object> -o solution
```

- `api.hpp` is provided by the grader, so `#include "api.hpp"` works. Don't paste a modified copy of the API into your file.
- The grader supplies `main`. If your file defines `main`, linking fails.
- Only the C and C++ standard libraries are available (no Boost or other third-party libraries). No other flags are added (in particular, no `-march`), so by default the code is built for generic x86-64.
- Compilation must finish within 60 seconds. If it fails, the submission shows **Compile error** with the compiler output, and it scores 0.

### Running

Your program runs in a sandbox:

- **Single-threaded.** You can't create threads or child processes. `std::thread`, `std::async` with `std::launch::async`, `fork` and similar calls fail (for example, `std::thread` throws `std::system_error`).
- **No file or network access.** Your program is the only file in its filesystem, so opening any file fails, and there is no network.
- **No output.** Anything you write to stdout or stderr is discarded, so printing debug output only costs you time. There is no input on stdin either.
- **Memory:** 2 GB of address space (virtual memory) for each test, including your code, stack and heap. An allocation beyond that fails.

### Timing

- **Setup:** `createSolution()` plus all the `registerOption()` calls get 2 seconds in total, which is not counted in your budget. If setup takes longer or crashes, that test scores 0 and shows **Killed** or **Crashed**.
- **Budget:** each test has a budget of **1 second**. Only the time spent inside `process()` counts, measured by the grader from outside your program for each call: from the moment an update is handed to your program until your valuations are returned. This includes a small fixed cost (roughly 0.1 µs per call) for passing data in and out, which is the same for every team.
- **Running out:** when the budget is used up, the test stops. **The score gained so far is kept**, the call that crossed the budget earns nothing, and the remaining inputs earn nothing. The test shows **Budget reached**.
- **Crashes:** if your program crashes, exits, or throws an exception out of `process()`, the test stops in the same way: the score gained so far is kept and the remaining inputs earn nothing. The test shows **Crashed**.
- Timings can vary slightly from run to run. A solution that only just finishes within the budget may get a slightly different score if you submit it again.

### Hardware

Submissions are graded on AWS `c7i.metal-24xl` machines with 4th generation Intel Xeon Scalable (Sapphire Rapids) processors. Each test runs on its own physical core: your program runs on one hyperthread, and the other hyperthread of the same core is used only by the grader feeding your program. Other submissions run on other cores at the same time.

## Sample interaction

```text
registerOption({id=1, underlyingId=100, strike=70, tte=0.05})
process({U, 74, 100}, [])           -> []
process({V, 0.52, 1}, [])           -> [{1, 5.80}]
process({U, 75, 100}, [{1, 25}])    -> [{1, 6.53}]
```

1. The grader registers an option with id 1, underlying 100, strike 70 and TTE 0.05.
2. The grader tells you that the latest price of underlying 100 is 74. You have not sent any valuations yet, so `previousScores` is empty.
3. You don't send any valuation yet, returning an empty list.
4. The grader tells you that the latest volatility of option 1 is 0.52. `previousScores` is still empty, because you returned nothing on the previous call.
5. You compute the Black–Scholes price from all the inputs, approximately `5.80`, and return `{{1, 5.80}}`.
6. The grader tells you that underlying 100 has moved to 75, and that your previous valuation scored 25 (an example value).
7. You recalculate the price of the option and return 6.53 for option 1.

## Submissions

- Submit on the **Submit** tab. Submit only the implementation `.cpp` file, without a `main` function, at most 64 KB. Don't submit the header or your test files.
- Each team can submit **once per minute**. The organisers may change this limit during the contest; the **Submit** tab always shows when your next submission is allowed.
- Your results appear on **My submissions** as soon as grading finishes, usually within a minute. For each test you can see the score, the time used out of the 1 second budget, how many inputs were processed, and the outcome (**Completed**, **Budget reached**, **Crashed** or **Killed**).
- Your best submission (marked ★) is the one that counts. A later, lower-scoring submission never lowers your best.
- The **Leaderboard** is live and shows each team's rank, name and best total. If two teams have the same best total, the team that submitted that score first ranks higher.
- If a submission shows **Grader error**, the problem is on our side and it is retried automatically. If it stays that way, tell the organisers.
- Submissions are accepted only while the contest is open. After it ends, the site stays readable but you can't submit.
- One login token is shared by your whole team. Don't share your token or login link with anyone outside your team.
