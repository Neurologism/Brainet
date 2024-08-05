#ifndef GRAPH_INCLUDE_GUARD
#define GRAPH_INCLUDE_GUARD

#include "dependencies.h"
#include "variable.h"
#include "operation/operation.h"

/**
 * @brief The graph class is a implementation of a computational graph. It is used to store the variables and operations and to execute the forward and backward pass.
 */
class GRAPH 
{
    std::vector<std::shared_ptr<VARIABLE>> __variables; // all variables in the graph
    std::vector<std::shared_ptr<VARIABLE>> __outputs; // the variables that are to be differentiated
    
    /**
     * @brief This function builds the gradient table for the variable focus. It is a recursive function that calculates the gradient of the focus variable with respect to all other variables in the graph.
     * To do this it uses dynamic programming.
     * @param focus The variable for which the gradient is calculated.
     * @param grad_table The gradient table that stores already calculated gradients.
     */
    void build_grad(std::shared_ptr<VARIABLE> focus, std::vector<std::shared_ptr<TENSOR<double>>> & grad_table);
    /**
     * @brief This function performs a topological sort on the graph and returns the sorted variables.
     * @return std::vector<std::shared_ptr<VARIABLE>> The sorted variables.
     */
    std::vector<std::shared_ptr<VARIABLE>> __topo_sort();
public:
    GRAPH() = default;
    ~GRAPH() = default;
    /**
     * @brief This function simply executes the operations of the graph in topological order.
     */
    void forward();
    /**
     * @brief This function calculates the gradients of the target variables with respect to the variables in the differentiate vector.
     * It utilizes the well know general backpropagation algorithm and is implemented in a dynamic programming fashion.
     * @param target The target variables for which the gradients are calculated.
     * @return std::vector<std::shared_ptr<TENSOR<double>> The gradients of the target variables with respect to the differentiate variables.
     */
    std::vector<std::shared_ptr<TENSOR<double>>> backprop(std::vector<std::shared_ptr<VARIABLE>> & target);
    /**
     * @brief This function returns all variables in the graph.
     * @return std::vector<std::shared_ptr<VARIABLE>> The variables in the graph.
     */
    std::vector<std::shared_ptr<VARIABLE>> get_variables();
    /**
     * @brief This function adds a variable to the graph.
     * @param var The variable to be added.
     * @return std::shared_ptr<VARIABLE> The added variable.
     */
    std::shared_ptr<VARIABLE> add_variable(std::shared_ptr<VARIABLE> var)
    {
        __variables.push_back(var); 
        if(var->get_operation()!=nullptr)var->get_operation()->set_variable(__variables.back()); // address of variable has changed -> invalidation of pointers
        return __variables.back();
    };
    
    /**
     * @brief This function adds a variable to the list of variables that have a gradient of 1.
     * @param var The variable to be added.
     * @return std::uint32_t The index of the added variable used to access it later.
     */
    std::uint32_t add_output(std::shared_ptr<VARIABLE> var)
    {
        __outputs.push_back(var);
    };
    /**
     * @brief This function returns the data of the output variable at the given index.
     * @param index The index of the output variable.
     * @return std::shared_ptr<TENSOR<double>> The data of the output variable.
     */
    std::shared_ptr<TENSOR<double>> get_output(std::uint32_t index)
    {
        return __outputs[index]->get_data();
    };
}; 


std::vector<std::shared_ptr<VARIABLE>> GRAPH::__topo_sort()
{
    std::vector<std::shared_ptr<VARIABLE>> sorted;
    std::vector<bool> visited(__variables.size(), false);
    // depth first search
    std::function<void(std::shared_ptr<VARIABLE>)> dfs = [&](std::shared_ptr<VARIABLE> var) 
    {
        if(visited[var->get_id()]) return;
        visited[var->get_id()] = true;
        for (std::shared_ptr<VARIABLE> child : var->get_consumers())
        {
            if (!visited[child->get_id()])
            {
                dfs(child);
            }
        }
        sorted.push_back(var);
    };

    for (std::shared_ptr<VARIABLE> var : __variables) // call for all variables
    {
        if (!visited[var->get_id()])
        {
            dfs(var);
        }
    }
    std::reverse(sorted.begin(), sorted.end()); // reverse the order to get the topological order
    return sorted;
}

void GRAPH::forward()
{
    std::vector<std::shared_ptr<VARIABLE>> sorted = __topo_sort(); // get the topological order
    for (std::shared_ptr<VARIABLE> var : sorted)
    {
        std::shared_ptr<OPERATION> op = var->get_operation();
        if (op != nullptr) // if the variable has an operation, execute it
        {
            std::vector<std::shared_ptr<VARIABLE>> inputs = var->get_inputs(); // get the inputs of the operation
            var->get_operation()->f(inputs); // execute the operation
        }
    }
}

std::vector<std::shared_ptr<TENSOR<double>>> GRAPH::backprop(std::vector<std::shared_ptr<VARIABLE>> & targets)
{
    std::vector<std::shared_ptr<TENSOR<double>>> grad_table(__variables.size(),nullptr); // data table for the gradients 
    for(std::shared_ptr<VARIABLE> var : __outputs) // initialize the gradient table with the output variables
    {
        grad_table[var->get_id()] = std::make_shared<TENSOR<double>>(TENSOR<double>(var->get_data()->shape())); // initialize the gradient table with the differentiate variables
        for(std::uint32_t i = 0; i < var->get_data()->size(); i++)
        {
            grad_table[var->get_id()]->data()[i] = 1; // differentiate variables have a gradient of 1 and are considered as leaf nodes
        }
    }

    for (std::shared_ptr<VARIABLE> var : targets)
    {
        build_grad(var, grad_table); // build the gradient table for the target variables
    }
    // only return the gradients of the target variables
    std::vector<std::shared_ptr<TENSOR<double>>> target_grads;
    for (std::shared_ptr<VARIABLE> var : targets)
    {
        target_grads.push_back(grad_table[var->get_id()]);
    }
    return target_grads;
}

void GRAPH::build_grad(std::shared_ptr<VARIABLE> focus, std::vector<std::shared_ptr<TENSOR<double>>> & grad_table)
{
    // error handling
    if (grad_table[focus->get_id()] != nullptr) // if the gradient has already been calculated, return
    {
        return;
    }
    if (focus->get_consumers().empty())
    {
        throw std::runtime_error("Variable has no consumers");
    }
    if (focus->get_data()->dimensionality() == 0)
    {
        throw std::runtime_error("Variable has no data");
    }
    
    std::shared_ptr<TENSOR<double>> _gradient;
    for (std::uint32_t i = 0; i < focus->get_consumers().size(); i++) // the sum of the gradients of the consumers is the gradient of the variable
    {
        // load stuff
        std::shared_ptr<VARIABLE> consumer = focus->get_consumers().at(i);
        std::shared_ptr<OPERATION> op = consumer->get_operation();
        build_grad(consumer, grad_table); // build the gradient table for the consumer (dp, dfs)
        std::shared_ptr<TENSOR<double>> gradient = op->bprop(consumer->get_inputs(), focus, grad_table[consumer->get_id()]); // calculate the gradient of the consumer with respect to the focus variable
        if (gradient->shape() != focus->get_data()->shape())
        {
            throw std::runtime_error("Gradient shape does not match variable shape");
        }
        if(_gradient == nullptr)
        {
            _gradient = gradient;
        }
        else if(_gradient->shape() != gradient->shape())
        {
            throw std::runtime_error("Gradient shapes do not match");
        }
        else
        {
            for (std::uint32_t j = 0; j < gradient->size(); j++) // add the gradient to the gradient table
            {
                _gradient->data()[j] += gradient->data()[j];
            }
        }
        
    }
    grad_table[focus->get_id()] = _gradient;
}

std::vector<std::shared_ptr<VARIABLE>> GRAPH::get_variables()
{
    return __variables;
}


#endif // GRAPH_INCLUDE_GUARD