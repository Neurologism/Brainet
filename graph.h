#ifndef GRAPH_INCLUDE_GUARD
#define GRAPH_INCLUDE_GUARD


#include "variable.h"

class GRAPH // dag of variables and operations
{
    std::vector<VARIABLE> __variables;
    std::vector<OPERATION> __operations;
public:
    GRAPH();
    ~GRAPH();
    VARIABLE * operator[](int index);

}; 

GRAPH::GRAPH()
{
}

GRAPH::~GRAPH()
{
}   

VARIABLE * GRAPH::operator[](int index)
{
    return &__variables[index];
}





#endif // GRAPH_INCLUDE_GUARD