#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include"..\operation.h"

/**
 * @brief ACTIVATION_FUNCTION class is an abstract class that defines the interface for all activation functions.
*/
template <typename T>
class ACTIVATION_FUNCTION : public OPERATION
{
public:
    ACTIVATION_FUNCTION(VARIABLE<T> * variable) : OPERATION(variable){};
    void f(std::vector<VARIABLE<T> *>& inputs) override;
    void bprop(std::vector<VARIABLE<T> *>& inputs, std::vector<VARIABLE<T> *> outputs) override;

    virtual double activation_function(double x) = 0;
    virtual double activation_function_derivative(double x) = 0;
};

template <typename T>
void ACTIVATION_FUNCTION<T>::f(std::vector<VARIABLE<T> *>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::f: Invalid number of input variables.");
    }
    
    DATATYPE data = inputs[0]->get_data()[0];
    data
}

template <typename T>
void ACTIVATION_FUNCTION<T>::bprop(std::vector<VARIABLE<T> *>& inputs, std::vector<VARIABLE<T> *> outputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::bprop: Invalid number of input variables.");
    }

    // load derivative of activation into tensor
    TENSOR _tensor;

    // sum the gradients of the outputs

    for (VARIABLE<T> * output : outputs)
    {
        TENSOR * tensor = output->get_tensor();
        for (int i = 0; i < tensor->size(); i++)
        {
            for (int j = 0; j < tensor->operator[](i)->size(); j++)
            {
                _tensor[i]->operator[](j)->set_data(_tensor[i]->operator[](j)->get_data() + tensor->operator[](i)->operator[](j)->get_data());
            }
        }
    }

    for (int i = 0; i < _tensor.size(); i++) // apply activation function derivative to all elements
    {
        for (int j = 0; i < _tensor[i]->size(); j++)
        {
            _tensor[i]->operator[](j)->set_data(activation_function_derivative(_tensor[i]->operator[](j)->get_data()));
        }
    }
}


#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD