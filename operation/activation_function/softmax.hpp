#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../operation.hpp"

/**
 * @brief Softmax function class, representing the softmax function f(x) = exp(x) / sum(exp(x)).
*/
class Softmax : public Operation
{   
protected:
    double activationFunction(double input);

    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;

public:
    Softmax() { mName = "SOFTMAX"; }
    ~Softmax() = default;
};

double Softmax::activationFunction(double input)
{
    return exp(input);
}



void Softmax::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Softmax::f: Invalid number of input variables.");
    }

    std::shared_ptr<Tensor<double>> _data = std::make_shared<Tensor<double>>(inputs.front()->getData()->shape()); // create a new tensor to store the result

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape()[0]; i++)
    {
        double _sum = 0;
        double _max = inputs.front()->getData()->at({i, 0}); // normalize the input to avoid overflow / underflow
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
        {
            _sum += activationFunction(inputs.front()->getData()->at({i, j}));
            if (inputs.front()->getData()->at({i, j}) > _max)
            {
                _max = inputs.front()->getData()->at({i, j});
            }
        }


        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)
        {

            _data->set({i, j}, activationFunction(inputs.front()->getData()->at({i, j}) - _max) / _sum);
        }
    }
    this->getVariable()->getData() = _data; // store the result in the variable
}

std::shared_ptr<Tensor<double>> Softmax::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    // always check for right number of inputs
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("Softmax::bprop: Invalid number of input variables.");
    }
    if (inputs.front() != focus)
    {
        throw std::invalid_argument("Softmax::bprop: The focus variable is not the input variable.");
    }

    std::shared_ptr<Tensor<double>> _data = this->getVariable()->getData(); // get the data of the variable
    std::shared_ptr<Tensor<double>> _grad = std::make_shared<Tensor<double>>(inputs.front()->getData()->shape()); // create a new tensor to store the gradient

    for (std::uint32_t i = 0; i < inputs.front()->getData()->shape()[0]; i++)
    {
        double _sum = 0;
        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++) // precalculate the sum of the gradient
        {
            _sum += _data->at({i, j}) * gradient->at({i, j});
        }

        for (std::uint32_t j = 0; j < inputs.front()->getData()->shape()[1]; j++)  
        {
            _sum -= _data->at({i, j}) * gradient->at({i, j});
            _grad->set({i, j}, _data->at({i,j}) * (1-_data->at({i,j})) * _grad->at({i,j}) - _sum * _data->at({i,j}));
            _sum += _data->at({i, j}) * gradient->at({i, j});
        }
    }

    return _grad; // return the gradient
}
#endif // SOFTMAX_HPP