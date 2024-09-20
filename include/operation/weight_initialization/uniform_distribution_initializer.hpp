#ifndef UNIFORM_DISTRIBUTION_INITIALIZER_HPP
#define UNIFORM_DISTRIBUTION_INITIALIZER_HPP

#include "weight_initializer.hpp"

/**
 * @brief Class to initialize a vector with random values from a uniform distribution.
 */
class UniformDistributionInitializer : public WeightInitializer
{
protected:

    double mLowerBound;
    double mUpperBound;
    std::uniform_real_distribution<double> mDist;

    double generate() override;

public:
    
    /**
     * @brief Create a random engine to generate random values.
     * @param mInputUnits The number of input units.
     * @param mOutputUnits The number of output units.
     */
    void createRandomEngine(std::uint32_t mInputUnits, std::uint32_t mOutputUnits) override;

    /**
    * @brief Construct a new Uniform Distribution to fill a vector with random values.
    * @param lowerBound The lower bound of the uniform distribution.
    * @param upperBound The upper bound of the uniform distribution.
    */
    UniformDistributionInitializer(double lowerBound, double upperBound);

    /**
     * @brief Construct a new Uniform Distribution Initializer object
     */
    UniformDistributionInitializer() = default;
};

#endif // UNIFORM_DISTRIBUTION_HPP