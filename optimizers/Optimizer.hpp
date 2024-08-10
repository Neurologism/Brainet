#ifndef Optimizer_HPP
#define Optimizer_HPP

#include "../dependencies.h"
#include "../graph.h"

/**
 * @brief The abstract class Optimizer is intended to be used as a base class for all optimization algorithms used to train the models.
 */
class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    virtual void update(const std::vector<std::shared_ptr<TENSOR<double>>> & gradients, std::uint32_t batch_size) = 0;
};


#include "primitive_SGD.hpp"

using Optimizer_Variant = std::variant<primitive_SGD>;


#endif // Optimizer_HPP