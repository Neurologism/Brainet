//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/he_initialization.hpp"

double HeInitialization::generate()
{
    return mDist(mGen);
}

void HeInitialization::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    mInputUnits = inputUnits;
    mOutputUnits = outputUnits;
    mLowerBound = -std::sqrt(6.0 / (inputUnits));
    mUpperBound = std::sqrt(6.0 / (outputUnits));
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<double>(mLowerBound, mUpperBound);
}