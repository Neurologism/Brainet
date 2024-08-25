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
     * @brief Construct a new He Initialization to fill a vector with random values.
     * @param inputUnits The number of input units.
     * @param outputUnits The number of output units.
     */
    HeInitialization(std::uint32_t inputUnits, std::uint32_t outputUnits);
};



#endif // HE_INITIALIZATION_HPP
