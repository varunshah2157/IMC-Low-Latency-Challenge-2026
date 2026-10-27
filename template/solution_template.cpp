#include "api.hpp"

class Solution : public ISolution {
public:
    void registerOption(const Option& option) override {

    }

    std::vector<Valuation> process(const ParameterUpdate& update, const std::vector<Score>& previousScores) override {
        return {};
    }
};

ISolution* createSolution() {
    return new Solution();
}