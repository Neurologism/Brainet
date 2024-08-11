#ifndef PRIMITIVE_SGD_HPP
#define PRIMITIVE_SGD_HPP

#include "Optimizer.hpp"
#include "../module/module.h"

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
    for(std::uint32_t i = 0; i < MODULE::get_learnable_parameters().size(); i++)
    {
        for(std::uint32_t j = 0; j < MODULE::get_learnable_parameters()[i]->get_data()->size(); j++)
        {
            MODULE::get_learnable_parameters()[i]->get_data()->data()[j] -= _initial_learning_rate * gradients[i]->data()[j] / batch_size;
        }
    }
    _initial_learning_rate *= _decay_rate; // primitive learning rate decay
}

#endif // PrimitiveSGD_HPP