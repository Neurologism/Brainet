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
    std::vector<std::vector<std::vector<double>>> backprop(std::vector<bool> & target, int z);
    void build_grad(int focus, std::vector<std::vector<double>> & grad_table);
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


/**
 * @brief backpropagation algorithm
 * @param target boolen list indicating for each variable in __variables if its gradient should be computed 
 * @param z the variable to be differentiated (gradient is 1)
*/
std::vector<std::vector<std::vector<double>>> GRAPH::backprop(std::vector<bool> & target, int z)
{
    std::vector<std::vector<double>> grad_table(__variables.size()); // data 
    grad_table[z] = {1};
    
}


void GRAPH::build_grad(int focus, std::vector<std::vector<double>> & grad_table)
{
    if (grad_table[focus].size() != 0)
    {
        return;
    }

    grad_table[focus].resize(__variables[focus].get_data().size());

    for (int i = 0; i < __variables[focus].get_consumers().size(); i++)
    {
        VARIABLE * consumer = __variables[focus].get_consumers()[i];
        OPERATION * op = consumer->get_operation();
        std::vector<VARIABLE *> inputs = consumer->get_inputs();
        build_grad(consumer->get_id(), grad_table);
        std::vector<double> gradient = op->bprop(consumer->get_inputs(), __variables[focus], grad_table[consumer->get_id()]);
        for (int j = 0; j < gradient.size(); j++)
        {
            grad_table[focus][j] += gradient[j];
        }
    }
}


#endif // GRAPH_INCLUDE_GUARD