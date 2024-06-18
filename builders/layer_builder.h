#ifndef LAYER_BUILDER_INCLUDE_GUARD
#define LAYER_BUILDER_INCLUDE_GUARD

#include "builder.h"
#include "../dependencies.h"
#include "../operation/activation_function/rectified_linear_unit.h"
#include "../operation/linear_algebra/matmul.h"

/**
 * @brief LAYER_BUILDER class is a builder class for creating a layer in a model.
*/
class LAYER_BUILDER : public BUILDER
{
public:
    LAYER_BUILDER(GRAPH * graph) : BUILDER(graph){};
    VARIABLE * add_linear_transformation(VARIABLE * head, std::vector<double> & weights, std::vector<int> & shape);
    VARIABLE * add_activation_function(VARIABLE * parent, OPERATION * activation_function);
    VARIABLE * add_input_layer(std::vector<double> & data, std::vector<int> & shape);
};

/**
 * @brief adds a linear transformation to the model
 * creates two VARIABLES, one to store the weights, which has no operation, and one to store the result of the matrix multiplication
 * it creates a MATMUL operation. 
 * @param parent the variable that should be the parent of the variable of the linear transformation
 * @param weights the weights used to initalize the pseudo-variable
 * @param shape the shape of the weight matrix
 * @return the new child that was added to the parent variable and represents the result of the matrix multiplication
*/
VARIABLE * LAYER_BUILDER::add_linear_transformation(VARIABLE * parent, std::vector<double> & weights, std::vector<int> & shape)
{
    __graph->add_variable(VARIABLE(nullptr, {}, {})); // nullptr because there is no operation
    VARIABLE * _weights = &__graph->get_variables().back();
    __graph->add_variable(VARIABLE(new MATMUL(), {parent, _weights}, {}));
    VARIABLE * _variable = &__graph->get_variables().back();
    _weights->set_data(weights);
    _weights->set_shape(shape);
    parent->get_consumers().push_back(_variable);
    _weights->get_consumers().push_back(_variable);
    
    return _variable;
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
    __graph->add_variable(VARIABLE(activation_function, {parent}, {}));
    VARIABLE * _variable = &__graph->get_variables().back();
    parent->get_consumers().push_back(_variable);
    return _variable;
}

/**
 * @brief adds an input layer to the model
 * creates a new VARIABLE with no operation and the input as the data
 * @param data the input data
 * @param shape the shape of the input data
 * @return the new child that was added to the parent variable and represents the input
*/
// think about adding a param for padding input 
VARIABLE * LAYER_BUILDER::add_input_layer(std::vector<double> & data, std::vector<int> & shape)
{
    __graph->add_variable(VARIABLE(nullptr, {}, {}));
    VARIABLE * _variable = &__graph->get_variables().back();
    _variable->set_data(data);
    _variable->set_shape(shape);
    return _variable;
}

#endif // LAYER_BUILDER_INCLUDE_GUARD