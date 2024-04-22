#ifndef ACTIVATION_FUNCTION_INCLUDE_GUARD
#define ACTIVATION_FUNCTION_INCLUDE_GUARD

#include <vector>
#include <stdexcept>
#include"operation.h"
#include"variable.h"

class ACTIVATION_FUNCTION : public OPERATION
{
public:
    ACTIVATION_FUNCTION(){};
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
    
    

    __shape = inputs.front()->get_operation()->get_shape(); // shape stays the same 

    for (double data : inputs.front()->get_operation()->get_data()) // apply activation function to all elements
    {
        __data.push_back(activation_function(data));
    }
}




#endif // ACTIVATION_FUNCTION_INCLUDE_GUARD