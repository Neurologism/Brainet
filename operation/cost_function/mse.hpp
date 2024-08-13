#ifndef MSE_HPP
#define MSE_HPP

#include"../operation.hpp"


/**
 * @brief Mean squared error class, representing the function f(x, y) = (1/n) * sum((x_i - y_i)^2) for i = 1 to n.
*/
class MSE : public Operation
{
public:
    MSE() { __dbg_name = "MSE"; }
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};



void MSE::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MSE operation requires 2 inputs");
    }
    if(inputs[0]->get_data()->shape() != inputs[1]->get_data()->shape())
    {
        throw std::runtime_error("MSE operation requires inputs to have the same shape");
    }
    if(inputs[0]->get_data()->dimensionality() != 2)
    {
        throw std::runtime_error("MSE::f: Other than 2D tensors are not supported");
    }

    // calculate the mean squared error
    double sum = 0;
    for(std::uint32_t i = 0; i < inputs[0]->get_data()->capacity(); i++)
    {
        sum += pow(inputs[0]->get_data()->at(i) - inputs[1]->get_data()->at(i), 2)/2;
    }
    sum /= inputs[0]->get_data()->capacity();
    // store the result
    this->get_variable()->get_data() = std::make_shared<Tensor<double>>(Tensor<double>({1},sum));
}


std::shared_ptr<Tensor<double>> MSE::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MSE operation requires 2 inputs");
    }
    if(inputs[0]->get_data()->shape() != inputs[1]->get_data()->shape())
    {
        throw std::runtime_error("MSE operation requires inputs to have the same shape");
    }
    if(gradient->shape() != std::vector<std::uint32_t>({1}))
    {
        throw std::runtime_error("MSE operation requires gradient to have shape {1}");
    }

    // calculate the gradient of the mean squared error function
    std::shared_ptr<Tensor<double>> _gradient = std::make_shared<Tensor<double>>(Tensor<double>(inputs[0]->get_data()->shape()));
    
    for(std::uint32_t i = 0; i < inputs[0]->get_data()->capacity(); i++)
    {
        _gradient->set(i, -(inputs[0]->get_data()->at(i) - inputs[1]->get_data()->at(i)) / inputs[0]->get_data()->shape(1)); // only divide by the size of 1 training example
    }

    return _gradient;
}

#endif // MSE_HPP