#ifndef GRAPH_INCLUDE_GUARD
#define GRAPH_INCLUDE_GUARD

#include "dependencies.h"
#include "variable.h"
#include "operation/operation.h"

class GRAPH // dag of variables and operations
{
    std::list<VARIABLE> __variables;
    void build_grad(int focus, std::vector<TENSOR<double>> & grad_table);
    std::list<VARIABLE *> __topo_sort();
public:
    GRAPH();
    ~GRAPH();
    VARIABLE * operator[](int index);
    void forward();
    std::vector<TENSOR<double>> backprop(std::vector<bool> & target, int z);
    std::list<VARIABLE> & get_variables();
    void add_variable(const VARIABLE &var){__variables.push_back(var);};
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
        if(visited[node]) return;
        visited[node] = true;
        for (auto children : *__variables[node].get_consumers())
        {
            if (!visited[children->get_id()])
            {
                dfs(children->get_id());
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
        OPERATION * op = sorted[i]->get_operation();
        if(op != nullptr)
        {
            op->f(*(sorted[i]->get_inputs()));
        }
        
    }
}


/**
 * @brief backpropagation algorithm
 * @attention assumes that the graph is a dag and forward has been called before
 * @param targets boolen list indicating for each variable in __variables if its gradient should be computed 
 * @param z the variable to be differentiated (gradient is 1)
*/
std::vector<TENSOR<double>> GRAPH::backprop(std::vector<bool> & targets, int z)
{
    std::vector<TENSOR<double>> grad_table(__variables.size()); // data 
    TENSOR<double> grad({1});
    grad.set({0},1);
    grad_table[z] = grad; // change to make possible to differentiate with respect to multiple variables ?

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
void GRAPH::build_grad(int focus, std::vector<TENSOR<double>> & grad_table)
{
    if (grad_table[focus].dimensionality() != 0) // gradient of this variable already computed
    {
        return;
    }

    grad_table[focus] = TENSOR<double>(__variables[focus].get_data()->shape()); // make space for the gradient

    for (int i = 0; i < __variables[focus].get_consumers()->size(); i++) // the sum of the gradients of the consumers is the gradient of the variable
    {
        // load stuff
        VARIABLE * consumer = __variables[focus].get_consumers()->at(i);
        OPERATION * op = consumer->get_operation();
        std::vector<VARIABLE *> inputs = *(consumer->get_inputs());
        build_grad(consumer->get_id(), grad_table); // build the gradient table for the consumer (dp, dfs)
        TENSOR<double> gradient = op->bprop(*(consumer->get_inputs()), __variables[focus], grad_table[consumer->get_id()]); // calculate the gradient of the consumer with respect to the focus variable
        for (int j = 0; j < gradient.size(); j++) // add the gradient to the gradient table
        {
            grad_table[focus].data()[j] += gradient.data()[j];
        }
    }
}

std::vector<VARIABLE> & GRAPH::get_variables()
{
    return __variables;
}


#endif // GRAPH_INCLUDE_GUARD