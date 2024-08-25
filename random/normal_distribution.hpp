#ifndef NORMAL_DISTRIBUTION_HPP
#define NORMAL_DISTRIBUTION_HPP

#include "random.hpp"

/**
 * @brief Class to initialize a vector with random values from a normal distribution.
 */
template <typename T>
class NormalDistribution : public Random<T>
{
    T mMean;
    T mStdDev;
    std::normal_distribution<T> mDist;

    T generate() override;

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
    NormalDistribution(T mean, T stdDev);
};

template <typename T>
T NormalDistribution<T>::generate()
{
    std::normal_distribution<T> dist(mMean, mStdDev);
    return dist(mGen);
}

template <typename T>
void NormalDistribution<T>::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    Random<T>::createRandomEngine(inputUnits, outputUnits);
    mGen = std::mt19937(mRd());
    mDist = std::normal_distribution<T>(mMean, mStdDev);
}

template <typename T>
NormalDistribution<T>::NormalDistribution(T mean, T stdDev)
{
    mMean = mean;
    mStdDev = stdDev;
}

#endif // NORMAL_DISTRIBUTION_HPP