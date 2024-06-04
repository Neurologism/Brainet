#ifndef LAYER_BUILDER_INCLUDE_GUARD
#define LAYER_BUILDER_INCLUDE_GUARD

#include "builder.h"
#include "../dependencies.h"
#include "../operation/activation_function/rectified_linear_unit.h"
#include "../operation/linear_algebra/matmul.h"
#include "../operation/void_operation.h"

/**
 * @brief LAYER_BUILDER class is a builder class for creating a layer in a model.
*/
class LAYER_BUILDER : public BUILDER
{
public:
    VARIABLE * add_linear_transformation(VARIABLE * head, std::vector<double> & weights, std::vector<int> & shape);
    VARIABLE * add_activation_function(VARIABLE * parent, OPERATION * activation_function);
};

/**
 * @brief adds a linear transformation to the model
 * creates two VARIABLES, one to store the weights, which is a pseudo-variable with a VOID_OPERATION, and one to store the result of the matrix multiplication
 * it creates a MATMUL operation. 
 * @param parent the variable that should be the parent of the variable of the linear transformation
 * @param weights the weights used to initalize the pseudo-variable
 * @param shape the shape of the weight matrix
 * @return the new child that was added to the parent variable and represents the result of the matrix multiplication
*/
VARIABLE * LAYER_BUILDER::add_linear_transformation(VARIABLE * parent,std::vector<double> & weights, std::vector<int> & shape)
{
    VARIABLE * _weights = new VARIABLE(new VOID_OPERATION(), {}, {});
    _weights->set_data(weights);
    _weights->set_shape(shape);
    __graph->add_variable(weights);
    VARIABLE * variable = new VARIABLE(new MATMUL(), {parent, _weights}, {});
    parent->get_consumers().push_back(variable);
    _weights->get_consumers().push_back(variable);
    __graph->add_variable(variable);
    return &VARIABLE;
}


/**
 * @brief adds an activation function to the model
 * creates a new VARIABLE with the activation function as the operation and the parent as the input
 * @param parent the variable that should be the parent of the variable of the activation function
 * @param activation_function the activation function that should be used
 * @return the new child that was added to the parent variable and represents the result of the activation function
*/
VARIABLE * LAYER_BUILDER::add_activation_function(VARIABLE * parent, OPERATION * activation_function)
{
    VARIABLE * variable = new VARIABLE(activation_function, {parent}, {});
    parent->get_consumers().push_back(variable);
    __graph->add_variable(variable);
    return variable;
}



#endif // LAYER_BUILDER_INCLUDE_GUARD