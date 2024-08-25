#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "../dependencies.hpp"

/**
 * @brief Base class to initialize a vector randomly.
 */
template <typename T>
class Random
{
protected:

    std::random_device mRd;
    std::mt19937 mGen;
    std::uint32_t mInputUnits;
    std::uint32_t mOutputUnits;

    virtual T generate() = 0;

public:

    virtual void createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits);

    /**
     * @brief Generate a vector of random values.
     * @param size The size of the vector.
     * @return The vector of random values.
     */
    std::vector<T> createRandomVector()
};

template <typename T>
void Random<T>::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    mInputUnits = inputUnits;
    mOutputUnits = outputUnits;
}


template <typename T>
std::vector<T> Random<T>::createRandomVector()
{
    std::vector<T> output(mInputUnits * mOutputUnits);

    for (std::uint32_t i = 0; i < mInputUnits * mOutputUnits; i++)
    {
        output[i] = generate_random();
    }

    return output;
}

#include "uniform_distribution.hpp"
#include "normal_distribution.hpp"
#include "normalized_initialization.hpp"
#include "he_initialization.hpp"

using RandomVariant = std::variant<UniformDistribution<double>, NormalDistribution<double>, NormalizedInitialization, HeInitialization>;

#endif // RANDOM_HPP