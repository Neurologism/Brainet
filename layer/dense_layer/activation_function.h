#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include"operation.h"
#include"variable.h"

class ACTIVATION_FUNCTION : public OPERATION
{
public:
    ACTIVATION_FUNCTION(VARIABLE * variable) : OPERATION(variable){};
    void f(std::vector<VARIABLE *>& inputs) override;
    void bprop(std::vector<VARIABLE *>& inputs, VARIABLE * focus, std::vector<VARIABLE *> outputs) override;

    virtual double activation_function(double x) = 0;
    virtual double activation_function_derivative(double x) = 0;
};

void ACTIVATION_FUNCTION::f(std::vector<VARIABLE *>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument("ACTIVATION_FUNCTION::f: Invalid number of input variables.");
    }
    
    std::vector<int> _shape = inputs.front()->get_shape(); // shape stays the same 
    std::vector<double> _data;

    for (double data : inputs.front()->get_data()) // apply activation function to all elements
    {
        _data.push_back(activation_function(data));
    }

    __variable->set_data(_data);
    __variable->set_shape(_shape);
}




#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD