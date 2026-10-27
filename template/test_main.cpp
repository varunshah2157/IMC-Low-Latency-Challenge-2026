#include "api.hpp"

#include <iostream>
#include <unordered_map>
#include <cmath>
#include <limits>

std::vector<Score> unknown_scoring_function(std::vector<Valuation> valuations) {
    std::vector<Score> scores{};
    for(const auto& val: valuations) {
        scores.emplace_back(val.optionId, 42.0);
    }
    return scores;
}

int main() {
    ISolution* solution = createSolution();
    solution->registerOption(Option{1, 100, 70, 0.05});
    std::vector<ParameterUpdate> updates{
        ParameterUpdate{'U', 74, 100},
        ParameterUpdate{'V', 0.52, 1},
        ParameterUpdate{'U', 75, 100}
    };
    long double total_score = 0.0;
    std::vector<Score> previousScores{};
    for (const auto& update: updates) {
        std::vector<Valuation> valuations = solution->process(update, previousScores);
        previousScores = unknown_scoring_function(valuations);
        for (auto score : previousScores) {
            total_score += score.profit;
        }
    }
    std::cout << "Total profit: " << total_score << std::endl;

    return 0;
}