#ifndef UNIFORM_DISTRIBUTION_HPP
#define UNIFORM_DISTRIBUTION_HPP

#include "random.hpp"

/**
 * @brief Class to initialize a vector with random values from a uniform distribution.
 */
template <typename T>
class UniformDistribution : public Random<T>
{
protected
    T mLowerBound;
    T mUpperBound;
    std::uniform_real_distribution<T> mDist;

    T generate() override;

public:
    
    void createRandomEngine(std::uint32_t mInputUnits, std::uint32_t mOutputUnits) override;

    /**
    * @brief Construct a new Uniform Distribution to fill a vector with random values.
    * @param lowerBound The lower bound of the uniform distribution.
    * @param upperBound The upper bound of the uniform distribution.
    */
    UniformDistribution(T lowerBound, T upperBound);
};

template <typename T>
T UniformDistribution<T>::generate()
{
    return mDist(mGen);
}

template <typename T>
void UniformDistribution<T>::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    Random<T>::createRandomEngine(inputUnits, outputUnits);
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<T>(mLowerBound, mUpperBound);
}

template <typename T>
UniformDistribution<T>::UniformDistribution(T lowerBound, T upperBound)
{
    mLowerBound = lowerBound;
    mUpperBound = upperBound;
}

#endif // UNIFORM_DISTRIBUTION_HPP