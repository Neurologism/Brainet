#ifndef HE_INITIALIZATION_HPP
#define HE_INITIALIZATION_HPP

#include "uniform_distribution.hpp"

/**
 * @brief Class to initialize a vector with random values from a He initialization.
 */
class HeInitialization : public UniformDistribution<double>
{
public:
    /**
     * @brief Create a random engine to generate random values.
     * @param inputUnits The number of input units.
     * @param outputUnits The number of output units.
     */
    void createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits) override;

    /**
     * @brief Construct a new He Initialization to fill a vector with random values.
     */
    HeInitialization() = default;
};

void HeInitialization::createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits)
{
    UniformDistribution<double>::createRandomEngine(inputUnits, outputUnits);
    mLowerBound = -std::sqrt(6.0 / (inputUnits));
    mUpperBound = std::sqrt(6.0 / (outputUnits));
    mGen = std::mt19937(mRd());
    mDist = std::uniform_real_distribution<double>(mLowerBound, mUpperBound);
}




#endif // HE_INITIALIZATION_HPP
