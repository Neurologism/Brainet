#ifndef PRIMITIVESGD_HPP
#define PRIMITIVESGD_HPP

#include "optimizer.hpp"
#include "../module/module.hpp"

/**
 * @brief The PrimitiveSGD class is a simple implementation of the stochastic gradient descent algorithm.
 */
class PrimitiveSGD : public Optimizer
{
    double _initial_learning_rate;
    double _decay_rate;

public:
    PrimitiveSGD(double initial_learning_rate, double decay_rate);
    ~PrimitiveSGD() = default;

    void update(const std::vector<std::shared_ptr<Tensor<double>>> & gradients, std::uint32_t batch_size) override;
};

PrimitiveSGD::PrimitiveSGD(double initial_learning_rate, double decay_rate)
{
    _initial_learning_rate = initial_learning_rate;
    _decay_rate = decay_rate;
}

void PrimitiveSGD::update(const std::vector<std::shared_ptr<Tensor<double>>> & gradients, std::uint32_t batch_size)
{
    for(std::uint32_t i = 0; i < Module::getLearnableParameters().size(); i++)
    {
        for(std::uint32_t j = 0; j < Module::getLearnableParameters()[i]->getData()->capacity(); j++)
        {
            Module::getLearnableParameters()[i]->getData()->subtract(j, _initial_learning_rate * gradients[i]->at(j) / batch_size);
        }
    }
    _initial_learning_rate *= _decay_rate; // primitive learning rate decay
}

#endif // PRIMITIVESGD_HPP