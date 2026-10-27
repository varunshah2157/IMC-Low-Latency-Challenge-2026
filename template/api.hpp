#pragma once
#include <vector>

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