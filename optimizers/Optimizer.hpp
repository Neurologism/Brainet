#ifndef Optimizer_HPP
#define Optimizer_HPP

#include "../dependencies.h"
#include "../graph.h"

/**
 * @brief The abstract class Optimizer is intended to be used as a base class for all optimization algorithms used to train the models.
 */
class Optimizer
{
protected:
    std::shared_ptr<GRAPH> graph;
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    virtual void update(const std::vector<std::shared_ptr<TENSOR<double>>> & gradients) = 0; 
};


#endif // Optimizer_HPP