#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "../dependencies.hpp"

/**
 * @brief Base class to initialize a vector randomly.
 */
template <typename T>
class Random
{

    virtual T generate_random() = 0;

public:

    /**
     * @brief Generate a vector of random values.
     * @param size The size of the vector.
     * @return The vector of random values.
     */
    std::vector<T> generate_random_vector(std::uint32_t size)
    {
        std::vector<T> output(size);

        for (std::uint32_t i = 0; i < size; i++)
        {
            output[i] = generate_random();
        }

        return output;
    }

};


#include "uniform_distribution.hpp"
#include "normal_distribution.hpp"

using RandomVariant = std::variant<UniformDistribution<double>, NormalDistribution<double>, NormalizedInitialization, HeInitialization>;

#endif // RANDOM_HPP