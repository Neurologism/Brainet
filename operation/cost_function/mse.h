#ifndef MSE_INCLUDE_GUARD
#define MSE_INCLUDE_GUARD

#include"../operation.h"


/**
 * @brief Mean squared error class, representing the function f(x, y) = (1/n) * sum((x_i - y_i)^2) for i = 1 to n.
*/
class MSE : public OPERATION
{
public:
    MSE() { __dbg_name = "MSE"; }
    void f(std::vector<std::shared_ptr<VARIABLE>>& inputs) override;
    std::shared_ptr<TENSOR<double>> bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient) override;
};



void MSE::f(std::vector<std::shared_ptr<VARIABLE>>& inputs)
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
    // calculate the mean squared error
    double sum = 0;
    for(std::uint32_t i = 0; i < inputs[0]->get_data()->size(); i++)
    {
        sum += pow(inputs[0]->get_data()->data()[i] - inputs[1]->get_data()->data()[i], 2)/2;
    }
    sum /= inputs[0]->get_data()->size();
    // store the result
    this->get_variable()->get_data() = std::make_shared<TENSOR<double>>(TENSOR<double>({1},sum));
}


std::shared_ptr<TENSOR<double>> MSE::bprop(std::vector<std::shared_ptr<VARIABLE>>& inputs, std::shared_ptr<VARIABLE> & focus, std::shared_ptr<TENSOR<double>> & gradient)
{
    // security checks
    if(inputs.size() != 2)
    {
        throw std::runtime_error("MSE operation requires 2 inputs");
    }
    if(inputs[0]->get_data()->size() != inputs[1]->get_data()->size())
    {
        throw std::runtime_error("MSE operation requires inputs to have the same shape");
    }
    if(gradient->shape() != std::vector<std::uint32_t>({1}))
    {
        throw std::runtime_error("MSE operation requires gradient to have shape {1}");
    }

    // calculate the gradient of the mean squared error function
    std::shared_ptr<TENSOR<double>> _gradient = std::make_shared<TENSOR<double>>(TENSOR<double>(inputs[0]->get_data()->shape()));
    for(std::uint32_t i = 0; i < inputs[0]->get_data()->size(); i++)
    {
        _gradient->data()[i] = -(inputs[0]->get_data()->data()[i] - inputs[1]->get_data()->data()[i]) * gradient->data()[0] / inputs[0]->get_data()->shape(1); // only divide by the size of 1 training example
    }

    return _gradient;
}

#endif // MSE_INCLUDE_GUARD