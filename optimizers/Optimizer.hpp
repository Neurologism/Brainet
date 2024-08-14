#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../dependencies.hpp"
#include "../graph.hpp"

/**
 * @brief The abstract class Optimizer is intended to be used as a base class for all optimization algorithms used to train the models.
 */
class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    virtual void update(const std::vector<std::shared_ptr<Tensor<double>>> & gradients, std::uint32_t batchSize) = 0;
};


#include "primitive_SGD.hpp"

using OptimizerVariant = std::variant<PrimitiveSGD>;


#endif // OPTIMIZER_HPP