#ifndef NORMALIZED_INITIALIZATION_HPP
#define NORMALIZED_INITIALIZATION_HPP

#include "uniform_distribution_initializer.hpp"

/**
 * @brief Class to initialize a vector with random values from a normalized initialization.
 */
class NormalizedInitialization : public UniformDistributionInitializer
{
public:
    /**
     * @brief Create a random engine to generate random values.
     * @param inputUnits The number of input units.
     * @param outputUnits The number of output units.
     */
    void createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits) override;

    /**
     * @brief Construct a new Normalized Initialization to fill a vector with random values.
     */
    NormalizedInitialization() = default;
};

void NormalizedInitialization::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    UniformDistributionInitializer::createRandomEngine(inputUnits, outputUnits);
    mLowerBound = -std::sqrt(6.0 / (inputUnits + outputUnits));
    mUpperBound = std::sqrt(6.0 / (inputUnits + outputUnits));
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<double>(mLowerBound, mUpperBound);
}

#endif // NORMALIZED_INITIALIZATION_HPP