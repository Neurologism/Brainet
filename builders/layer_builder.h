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
protected:
    static VARIABLE * __end_of_stream=nullptr;
    
public:
    void add_matrix_multiplication();
    void add_activation_function();
};

void LAYER_BUILDER::add_matrix_multiplication()
{
    if(__end_of_stream!=nullptr) throw std::exception("End of stream is not set.");
    VARIABLE * variable = new VARIABLE(new MATMUL(), {__end_of_stream}, {});
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