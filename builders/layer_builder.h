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
protected:
    static VARIABLE * __end_of_stream=nullptr;
    
public:
    void add_matrix_multiplication(std::vector<double> & weights, std::vector<int> & shape);
    void add_activation_function(OPERATION * activation_function);
};

void LAYER_BUILDER::add_matrix_multiplication(std::vector<double> & weights, std::vector<int> & shape)
{
    if(__end_of_stream!=nullptr) throw std::exception("End of stream is not set.");
    VARIABLE * weights = new VARIABLE(new VOID_OPERATION(), {}, {});
    weights->set_data(weights);
    weights->set_shape(shape);
    __graph->add_variable(weights);
    VARIABLE * variable = new VARIABLE(new MATMUL(), {__end_of_stream, weights}, {});
    __end_of_stream->get_consumers().push_back(variable);
    weights->get_consumers().push_back(variable);
    __graph->add_variable(variable);
    __end_of_stream = variable;
}



void LAYER_BUILDER::add_activation_function(OPERATION * activation_function)
{
    if(__end_of_stream!=nullptr) throw std::exception("End of stream is not set.");
    VARIABLE * variable = new VARIABLE(activation_function, {__end_of_stream}, {});
    __end_of_stream->get_consumers().push_back(variable);
    __graph->add_variable(variable);
    __end_of_stream = variable;
}

#endif // LAYER_BUILDER_INCLUDE_GUARD