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

#endif //WEIGHT_MATRIX_INITIALIZER_HPP
