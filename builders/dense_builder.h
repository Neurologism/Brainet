#ifndef DENSE_BUILDER_INCLUDE_GUARD
#define DENSE_BUILDER_INCLUDE_GUARD

#include "builder.h"
#include "../operation/activation_function/rectified_linear_unit.h"

/**
 * @brief DENSE_BUILDER class is a builder class for creating a dense layer in a model.
*/
class DENSE_BUILDER : public BUILDER
{

public:
    void add_relu();
};

void DENSE_BUILDER::add_relu()
{
    ReLU * relu = new ReLU();
    VARIABLE * variable = new VARIABLE(relu, {__end_of_stream}, {});
    __end_of_stream->get_consumers().push_back(variable);
    __graph->add_variable(variable);
    __end_of_stream = variable;
}

#endif // DENSE_BUILDER_INCLUDE_GUARD