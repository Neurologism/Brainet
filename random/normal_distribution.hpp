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

    T generate_random() override;

public:

    /**
     * @brief Construct a new Normal Distribution to fill a vector with random values.
     * @param mean The mean of the normal distribution.
     * @param stdDev The standard deviation of the normal distribution.
     */
    NormalDistribution(T mean, T stdDev);
};

template <typename T>
T NormalDistribution<T>::generate_random()
{
    std::normal_distribution<T> dist(mMean, mStdDev);
    return dist(mGen);
}

template <typename T>
NormalDistribution<T>::NormalDistribution(T mean, T stdDev)
{
    mGen = std::mt19937(mRd());
    mMean = mean;
    mStdDev = stdDev;
}

#endif // NORMAL_DISTRIBUTION_HPP