#ifndef DENSE_BUILDER_INCLUDE_GUARD
#define DENSE_BUILDER_INCLUDE_GUARD

#include "builder.h"

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
    
}

#endif // DENSE_BUILDER_INCLUDE_GUARD