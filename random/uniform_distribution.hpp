#ifndef UNIFORM_DISTRIBUTION_HPP
#define UNIFORM_DISTRIBUTION_HPP

#include "random.hpp"

/**
 * @brief Class to initialize a vector with random values from a uniform distribution.
 */
template <typename T>
class UniformDistribution : public Random<T>
{
    T mLowerBound;
    T mUpperBound;

    T generate() override;

public:
    
    void createRandomEngine(std::uint32_t mInputUnits, std::uint32_t mOutputUnits);

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
    std::uniform_real_distribution<T> dist(mLowerBound, mUpperBound);
    return dist(mGen);
}

template <typename T>
UniformDistribution<T>::UniformDistribution(T lowerBound, T upperBound)
{
    mGen = std::mt19937(mRd());
    mLowerBound = lowerBound;
    mUpperBound = upperBound;
}

class NormalizedInitialization
{

};

class HeInitialization
{

};

#endif // UNIFORM_DISTRIBUTION_HPP