#ifndef NORMAL_DISTRIBUTION_INITIALIZER_HPP
#define NORMAL_DISTRIBUTION_INITIALIZER_HPP

#include "weight_initializer.hpp"

/**
 * @brief Class to initialize a vector with random values from a normal distribution.
 */
class NormalDistributionInitializer : public WeightInitializer
{
    double mMean;
    double mStdDev;
    std::normal_distribution<double> mDist;

    double generate() override;

public:
    /**
     * @brief Create a random engine to generate random values.
     * @param inputUnits The number of input units.
     * @param outputUnits The number of output units.
     */
    void createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits) override;

    /**
     * @brief Construct a new Normal Distribution to fill a vector with random values.
     * @param mean The mean of the normal distribution.
     * @param stdDev The standard deviation of the normal distribution.
     */
    NormalDistributionInitializer(double mean, double stdDev);
};

double NormalDistributionInitializer::generate()
{
    std::normal_distribution<double> dist(mMean, mStdDev);
    return dist(mGen);
}

void NormalDistributionInitializer::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    WeightInitializer::createRandomEngine(inputUnits, outputUnits);
    mGen = std::mt19937(mRd());
    mDist = std::normal_distribution<double>(mMean, mStdDev);
}

NormalDistributionInitializer::NormalDistributionInitializer(double mean, double stdDev)
{
    mMean = mean;
    mStdDev = stdDev;
}

#endif // NORMAL_DISTRIBUTION_INITIALIZER_HPP