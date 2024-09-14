//
// Created by servant-of-scietia on 14.09.24.
//

#ifndef WEIGHT_MATRIX_INITIALIZER_HPP
#define WEIGHT_MATRIX_INITIALIZER_HPP

#include "../operation.hpp"
#include "weight_initializer.hpp"
#include "../activation_function/activation_function.hpp"

/**
  * @brief The WeightMatrixInitializer class is used to initialize a matrix in the first forward pass assuming the operation of the child is a matrix multiplication.
  */
class WeightMatrixInitializer : public Operation
{
	std::shared_ptr<WeightInitializer> mpWeightInitializer; // used to initialize the weight matrix
    double mBias; // the bias of used for the initialization
    std::uint32_t mM; // the number of columns in the weight matrix

    void createWeightMatrix(std::uint32_t n, std::uint32_t m);
public:

    explicit WeightMatrixInitializer(const std::uint32_t &m, std::shared_ptr<WeightInitializer> weightInitializer = std::make_shared<NormalizedInitialization>(), const double bias = 0) : mpWeightInitializer(std::move(weightInitializer)), mBias(bias), mM(m)
    {
        mName = "WeightMatrixInitializer";
    }

  	void f(std::vector<std::shared_ptr<Variable>> &inputs) override;

  	std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;


};

inline void WeightMatrixInitializer::createWeightMatrix(std::uint32_t n, std::uint32_t m)
{
    getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({n+1, m})); // initialize the weights randomly

    // initialize the weights randomly
    mpWeightInitializer->createRandomEngine(n, m);
    const std::vector<double> weights = mpWeightInitializer->createRandomVector();

    for (std::uint32_t i = 0; i < n; i++) // load the weights into the weight matrix
    {
        for (std::uint32_t j = 0; j < m; j++)
        {
            getVariable()->getData()->set({i, j}, weights[i*m+j]);
        }
    }

    for (std::uint32_t j = 0; j < m; j++)
    {
        getVariable()->getData()->set({n, j}, mBias);
    }
}

inline void WeightMatrixInitializer::f(std::vector<std::shared_ptr<Variable>> &inputs)
{
    // deduce the number of rows in the weight matrix
    const std::uint32_t n = getVariable()->getConsumers()[0]->getInputs()[0]->getData()->shape(1);

    createWeightMatrix(n, mM); // create the weight matrix
}

inline std::shared_ptr<Tensor<double>> WeightMatrixInitializer::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    throw std::runtime_error("WeightMatrixInitializer::bprop: This function should never be called.");
}




#endif //WEIGHT_MATRIX_INITIALIZER_HPP
