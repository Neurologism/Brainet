#ifndef HE_INITIALIZATION_HPP
#define HE_INITIALIZATION_HPP

#include "uniform_distribution_initializer.hpp"

/**
 * @brief Class to initialize a vector with random values from a He initialization.
 */
class HeInitialization : public UniformDistributionInitializer
{
    double generate() override;
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

#endif // HE_INITIALIZATION_HPP
