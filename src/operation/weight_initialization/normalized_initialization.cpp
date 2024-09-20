//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/normalized_initialization.hpp"

double NormalizedInitialization::generate()
{
    return mDist(mGen);
}

void NormalizedInitialization::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    mInputUnits = inputUnits;
    mOutputUnits = outputUnits;
    mLowerBound = -std::sqrt(6.0 / (inputUnits + outputUnits));
    mUpperBound = std::sqrt(6.0 / (inputUnits + outputUnits));
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<double>(mLowerBound, mUpperBound);
}