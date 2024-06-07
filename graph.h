#ifndef GRAPH_INCLUDE_GUARD
#define GRAPH_INCLUDE_GUARD

#include "dependencies.h"
#include "variable.h"

class GRAPH // dag of variables and operations
{
    std::vector<VARIABLE> __variables;
    void build_grad(int focus, std::vector<std::vector<double>> & grad_table);
    std::vector<VARIABLE *> __topo_sort();
public:
    GRAPH();
    ~GRAPH();
    VARIABLE * operator[](int index);
    void forward();
    std::vector<std::vector<double>> backprop(std::vector<bool> & target, int z);
    std::vector<VARIABLE> & get_variables();
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
 * @brief topological sort of the graph
 * @return Returns a list of pointers to the variables in topological order
*/
std::vector<VARIABLE *> GRAPH::__topo_sort()
{
    std::vector<VARIABLE *> sorted;
    std::vector<bool> visited(__variables.size(), false);
    std::function<void(int)> dfs = [&](int node)
    {
        visited[node] = true;
        for (int i = 0; i < __variables[node].get_consumers().size(); i++)
        {
            if (!visited[__variables[node].get_consumers()[i]->get_id()])
            {
                dfs(__variables[node].get_consumers()[i]->get_id());
            }
        }
        sorted.push_back(&__variables[node]);
    };

    for (int i = 0; i < __variables.size(); i++)
    {
        if (!visited[i])
        {
            dfs(i);
        }
    }
    std::reverse(sorted.begin(), sorted.end());
    return sorted;
}

/**
 * @brief algorithm executing the forward pass
 * @return Returns the value of all variables 
*/
void GRAPH::forward()
{
    std::vector<VARIABLE *> sorted = __topo_sort();
    for (int i = 0; i < sorted.size(); i++)
    {
        sorted[i]->get_operation()->f(sorted[i]->get_inputs());
    }
}


/**
 * @brief backpropagation algorithm
 * @attention assumes that the graph is a dag and forward has been called before
 * @param targets boolen list indicating for each variable in __variables if its gradient should be computed 
 * @param z the variable to be differentiated (gradient is 1)
*/
std::vector<std::vector<double>> GRAPH::backprop(std::vector<bool> & targets, int z)
{
    std::vector<std::vector<double>> grad_table(__variables.size()); // data 
    grad_table[z] = {1}; // change to make possible to differentiate with respect to multiple variables ?

    for (int i = 0; i < __variables.size(); i++)
    {
        if (targets[i]) // call build grad for each target variable
        {
            build_grad(i, grad_table);
        }
    }
    return grad_table;
}

/**
 * @brief builds the gradient table for the variable with id focus using dp with dfs 
 * @param focus the variable to be differentiated
 * @param grad_table the gradient table to be built
*/
void GRAPH::build_grad(int focus, std::vector<std::vector<double>> & grad_table)
{
    if (grad_table[focus].size() != 0) // gradient of this variable already computed
    {
        return;
    }

    grad_table[focus].resize(__variables[focus].get_data().size()); // make space for the gradient

    for (int i = 0; i < __variables[focus].get_consumers().size(); i++) // the sum of the gradients of the consumers is the gradient of the variable
    {
        // load stuff
        VARIABLE * consumer = __variables[focus].get_consumers()[i];
        OPERATION * op = consumer->get_operation();
        std::vector<VARIABLE *> inputs = consumer->get_inputs();
        build_grad(consumer->get_id(), grad_table); // build the gradient table for the consumer (dp, dfs)
        std::vector<double> gradient = op->bprop(consumer->get_inputs(), __variables[focus], grad_table[consumer->get_id()]); // calculate the gradient of the consumer with respect to the focus variable
        for (int j = 0; j < gradient.size(); j++) // add the gradient to the gradient table
        {
            grad_table[focus][j] += gradient[j];
        }
    }
}

std::vector<VARIABLE> & GRAPH::get_variables()
{
    return __variables;
}


#endif // GRAPH_INCLUDE_GUARD