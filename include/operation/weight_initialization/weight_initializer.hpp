#ifndef WEIGHT_INITIALIZER_HPP
#define WEIGHT_INITIALIZER_HPP

#include "../../dependencies.hpp"

/**
 * @brief Base class to initialize a vector randomly.
 */
class WeightInitializer
{
protected:

    std::random_device mRd;
    std::mt19937 mGen;
    std::uint32_t mInputUnits;
    std::uint32_t mOutputUnits;

    virtual double generate() = 0;

public:

    virtual void createRandomEngine(std::uint32_t inputUnits, std::uint32_t outputUnits) = 0;

    /**
     * @brief Generate a vector of random values.
     * @param size The size of the vector.
     * @return The vector of random values.
     */
    std::vector<double> createRandomVector();
};

#include "uniform_distribution_initializer.hpp"
#include "normal_distribution_initializer.hpp"
#include "normalized_initialization.hpp"
#include "he_initialization.hpp"

using InitializationVariant = std::variant<UniformDistributionInitializer, NormalDistributionInitializer, NormalizedInitialization, HeInitialization>;

#endif // WEIGHT_INITIALIZER_HPP