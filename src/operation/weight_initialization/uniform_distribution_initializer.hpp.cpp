//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/uniform_distribution_initializer.hpp"

double UniformDistributionInitializer::generate()
{
    return mDist(mGen);
}

void UniformDistributionInitializer::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    mInputUnits = inputUnits;
    mOutputUnits = outputUnits;
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<double>(mLowerBound, mUpperBound);
}

UniformDistributionInitializer::UniformDistributionInitializer(double lowerBound, double upperBound)
{
    mLowerBound = lowerBound;
    mUpperBound = upperBound;
}