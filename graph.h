#ifndef GRAPH_INCLUDE_GUARD
#define GRAPH_INCLUDE_GUARD

#include "dependencies.h"
#include "variable.h"
#include "operation/operation.h"

class GRAPH // dag of variables and operations
{
    std::list<VARIABLE> __variables;
    void build_grad(VARIABLE * focus, std::map<VARIABLE*,TENSOR<double>> & grad_table);
    std::list<VARIABLE *> __topo_sort();
public:
    GRAPH();
    ~GRAPH();
    void forward();
    std::map<VARIABLE*,TENSOR<double>> backprop(std::set<VARIABLE*> & target, std::vector<VARIABLE *> differentiate);
    std::list<VARIABLE> & get_variables();
    VARIABLE * add_variable(VARIABLE var)
    {
        __variables.push_back(var); 
        if(var.get_operation()!=nullptr)var.get_operation()->set_variable(&__variables.back()); // address of variable has changed -> invalidation of pointers
        return &__variables.back();
    };
}; 

GRAPH::GRAPH() 
{
}

GRAPH::~GRAPH()
{
    std::cout << "GRAPH DESTRUCTOR" << std::endl;
}   

/**
 * @brief topological sort of the graph
 * @return Returns a list of pointers to the variables in topological order
*/
std::list<VARIABLE *> GRAPH::__topo_sort()
{
    std::list<VARIABLE *> sorted;
    std::vector<bool> visited(__variables.size(), false);
    std::function<void(VARIABLE *)> dfs = [&](VARIABLE * var)
    {
        if(visited[var->get_id()]) return;
        visited[var->get_id()] = true;
        for (VARIABLE * child : *(var->get_consumers()))
        {
            if (!visited[child->get_id()])
            {
                dfs(child);
            }
        }
        sorted.push_back(var);
    };

    for (VARIABLE & var : __variables)
    {
        if (!visited[var.get_id()])
        {
            dfs(&var);
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
    std::list<VARIABLE *> sorted = __topo_sort();
    for (VARIABLE * var : sorted)
    {
        OPERATION * op = var->get_operation();
        if(op != nullptr)
        {
            op->f(*(var->get_inputs()));
        }
        
    }
}


/**
 * @brief backpropagation algorithm
 * @attention assumes that the graph is a dag and forward has been called before
 * @param targets boolen list indicating for each variable in __variables if its gradient should be computed 
 * @param differentiate the variables to be differentiated (gradient is 1)
*/
std::map<VARIABLE*,TENSOR<double>> GRAPH::backprop(std::set<VARIABLE*> & targets, std::vector<VARIABLE *> differentiate)
{
    std::map<VARIABLE*,TENSOR<double>> grad_table; // data 
    for(VARIABLE * var : differentiate)
    {
        grad_table[var] = TENSOR<double>(var->get_data()->shape());
        for(int i = 0; i < var->get_data()->size(); i++)
        {
            grad_table[var].data()[i] = 1;
        }
    }

    for (VARIABLE & var : __variables)
    {
        if (targets.find(&var)!=targets.end()) // call build grad for each target variable
        {
            build_grad(&var, grad_table);
        }
    }
    return grad_table;
}

/**
 * @brief builds the gradient table for the variable with id focus using dp with dfs 
 * @param focus the variable to be differentiated
 * @param grad_table the gradient table to be built
*/
void GRAPH::build_grad(VARIABLE * focus, std::map<VARIABLE*,TENSOR<double>> & grad_table)
{
    if (grad_table.find(focus)!=grad_table.end()) // gradient of this variable already computed
    {
        return;
    }

    grad_table[focus] = TENSOR<double>(focus->get_data()->shape()); // make space for the gradient

    for (int i = 0; i < focus->get_consumers()->size(); i++) // the sum of the gradients of the consumers is the gradient of the variable
    {
        // load stuff
        VARIABLE * consumer = focus->get_consumers()->at(i);
        OPERATION * op = consumer->get_operation();
        std::vector<VARIABLE *> inputs = *(consumer->get_inputs());
        build_grad(consumer, grad_table); // build the gradient table for the consumer (dp, dfs)
        TENSOR<double> gradient = op->bprop(*(consumer->get_inputs()), *focus, grad_table[consumer]); // calculate the gradient of the consumer with respect to the focus variable
        if (gradient.shape() != focus->get_data()->shape())
        {
            throw std::runtime_error("Gradient shape does not match variable shape");
        }
        for (int j = 0; j < gradient.size(); j++) // add the gradient to the gradient table
        {
            grad_table[focus].data()[j] += gradient.data()[j];
        }
    }
}

std::list<VARIABLE> & GRAPH::get_variables()
{
    return __variables;
}


#endif // GRAPH_INCLUDE_GUARD