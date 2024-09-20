//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/normal_distribution_initializer.hpp"

double NormalDistributionInitializer::generate()
{
    std::normal_distribution<double> dist(mMean, mStdDev);
    return dist(mGen);
}

void NormalDistributionInitializer::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    mInputUnits = inputUnits;
    mOutputUnits = outputUnits;
    mGen = std::mt19937(mRd());
    mDist = std::normal_distribution<double>(mMean, mStdDev);
}

NormalDistributionInitializer::NormalDistributionInitializer(double mean, double stdDev)
{
    mMean = mean;
    mStdDev = stdDev;
}