//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/weight_matrix_initializer.hpp"

void WeightMatrixInitializer::createWeightMatrix(std::uint32_t n, std::uint32_t m)
{
    getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({n, m})); // initialize the weights randomly

    // initialize the weights randomly
    mpWeightInitializer->createRandomEngine(n-1, m);
    const std::vector<double> weights = mpWeightInitializer->createRandomVector();

    for (std::uint32_t i = 0; i < n-1; i++) // load the weights into the weight matrix
    {
        for (std::uint32_t j = 0; j < m; j++)
        {
            getVariable()->getData()->set({i, j}, weights[i*m+j]);
        }
    }

    for (std::uint32_t j = 0; j < m; j++)
    {
        getVariable()->getData()->set({n-1, j}, mBias);
    }
}

void WeightMatrixInitializer::f(std::vector<std::shared_ptr<Variable>> &inputs)
{
    // deduce the number of rows in the weight matrix
    const std::uint32_t n = inputs[0]->getData()->shape(1);

    createWeightMatrix(n, mM); // create the weight matrix

    Variable::disconnectVariables(inputs[0], getVariable()); // one time use only
    getVariable()->setOperation(nullptr); // one time use only
}

std::shared_ptr<Tensor<double>> WeightMatrixInitializer::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    throw std::runtime_error("WeightMatrixInitializer::bprop: This function should never be called.");
}