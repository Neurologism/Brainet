#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "dependencies.hpp"
#include "variable.hpp"
#include "operation/operation.hpp"

/**
 * @brief The graph class is a implementation of a computational graph. It is used to store the variables and operations and to execute the forward and backward pass.
 */
class Graph 
{
    std::vector<std::shared_ptr<Variable>> mVariableVec; // all variables in the graph
    std::vector<std::shared_ptr<Tensor<double>>> mGradTable; // the gradient table for the variables
    
    /**
     * @brief This function builds the gradient table for the variable focus. It is a recursive function that calculates the gradient of the focus variable with respect to all other variables in the graph.
     * To do this it uses dynamic programming.
     * @param focus The variable for which the gradient is calculated.
     * @param gradTable The gradient table that stores already calculated gradients.
     */
    void mBuildGrad(std::shared_ptr<Variable> pFocus, std::vector<std::shared_ptr<Tensor<double>>> & pGradTable);
    /**
     * @brief This function performs a topological sort on the graph and returns the sorted variables.
     * @return std::vector<std::shared_ptr<Variable>> The sorted variables.
     */
    std::vector<std::shared_ptr<Variable>> mTopoSort();

public:
    Graph() = default;
    ~Graph() = default;

    /**
     * @brief This function simply executes the operations of the graph in topological order.
     */
    void forward();

    /**
     * @brief This function calculates the gradients of the target variables with respect to the variables in the differentiate vector.
     * It utilizes the well know general backpropagation algorithm and is implemented in a dynamic programming fashion.
     * @param targetVariables The variables for which the gradients are calculated.
     * @param outputVariables The target variables are computed with respect to the output variables.
     * @return The gradients of the target variables in the same order as the target variables.
     */
    void backprop(std::vector<std::shared_ptr<Variable>> & targetVariables, std::vector<std::shared_ptr<Variable>> & outputVariables);

    /**
     * @brief This function returns all variables in the graph.
     * @return std::vector<std::shared_ptr<Variable>> The variables in the graph.
     */
    std::vector<std::shared_ptr<Variable>> getVariableVec();

    /**
     * @brief This function adds a variable to the graph.
     * @param var The variable to be added.
     * @return std::shared_ptr<Variable> The added variable.
     */
    std::shared_ptr<Variable> addVariable(const std::shared_ptr<Variable> & pVar)
    {
        mVariableVec.push_back(pVar); 
        if(pVar->getOperation()!=nullptr)pVar->getOperation()->setVariable(mVariableVec.back()); // address of variable has changed -> invalidation of pointers; not nice but works
        return mVariableVec.back();
    };
}; 


std::vector<std::shared_ptr<Variable>> Graph::mTopoSort()
{
    std::vector<std::shared_ptr<Variable>> pSorted;
    std::vector<bool> visited(mVariableVec.size(), false);
    // depth first search
    std::function<void(std::shared_ptr<Variable>)> dfs = [&](std::shared_ptr<Variable> pVar) 
    {
        if(visited[pVar->getId()]) return;
        visited[pVar->getId()] = true;
        for (std::shared_ptr<Variable> pChild : pVar->getConsumers())
        {
            if (!visited[pChild->getId()])
            {
                dfs(pChild);
            }
        }
        pSorted.push_back(pVar);
    };

    for (std::shared_ptr<Variable> pVar : mVariableVec) // call for all variables
    {
        if (!visited[pVar->getId()])
        {
            dfs(pVar);
        }
    }
    std::reverse(pSorted.begin(), pSorted.end()); // reverse the order to get the topological order
    return pSorted;
}

void Graph::forward()
{
    std::vector<std::shared_ptr<Variable>> pSorted = mTopoSort(); // get the topological order
    for (std::shared_ptr<Variable> var : pSorted)
    {
        if (var->getOperation() != nullptr) // if the variable has an operation, execute it
        {
            var->getOperation()->f(var->getInputs()); // execute the operation
        }
    }
}

void Graph::backprop(std::vector<std::shared_ptr<Variable>> & targetVariables, std::vector<std::shared_ptr<Variable>> & outputVariables)
{
    std::vector<std::shared_ptr<Tensor<double>>> gradTable(mVariableVec.size(),nullptr); // data table for the gradients 
    for(std::shared_ptr<Variable> pVar : outputVariables) // initialize the gradient table with the output variables
    {
        gradTable[pVar->getId()] = std::make_shared<Tensor<double>>(Tensor<double>(pVar->getData()->shape())); // initialize the gradient table with the differentiate variables
        for(std::uint32_t i = 0; i < pVar->getData()->capacity(); i++)
        {
            gradTable[pVar->getId()]->set(i, 1); // differentiate variables have a gradient of 1 and are considered as leaf nodes
        }
    }

    std::vector<std::shared_ptr<Tensor<double>>> targetGradTable;

    for (std::shared_ptr<Variable> pVar : targetVariables) // calculate the gradients for all target variables
    {
        mBuildGrad(pVar, gradTable); // build the gradient table for the target variables
        targetGradTable.push_back(gradTable[pVar->getId()]);
    }

    return targetGradTable;
}

void Graph::mBuildGrad(std::shared_ptr<Variable> pFocus, std::vector<std::shared_ptr<Tensor<double>>> & gradTable)
{
    // error handling
    if (gradTable[pFocus->getId()] != nullptr) // if the gradient has already been calculated, return
    {
        return;
    }
    if (pFocus->getConsumers().empty())
    {
        throw std::runtime_error("Variable has no consumers");
    }
    if (pFocus->getData()->dimensionality() == 0)
    {
        throw std::runtime_error("Variable has no data");
    }
    
    std::shared_ptr<Tensor<double>> pGradient;
    for (std::uint32_t i = 0; i < pFocus->getConsumers().size(); i++) // the sum of the gradients of the consumers is the gradient of the variable
    {
        // load stuff
        std::shared_ptr<Variable> pConsumer = pFocus->getConsumers().at(i);
        std::shared_ptr<Operation> pOperation = pConsumer->getOperation();
        mBuildGrad(pConsumer, gradTable); // build the gradient table for the consumer
        std::shared_ptr<Tensor<double>> pGradientPart = pOperation->bprop(pConsumer->getInputs(), pFocus, gradTable[pConsumer->getId()]); // calculate the gradient of the consumer with respect to the focus variable
        if (pGradientPart->shape() != pFocus->getData()->shape())
        {
            throw std::runtime_error("Gradient shape does not match variable shape");
        }
        if(pGradient == nullptr)
        {
            pGradient = pGradientPart;
        }
        else if(pGradient->shape() != pGradientPart->shape())
        {
            throw std::runtime_error("Gradient shapes do not match");
        }
        else
        {
            for (std::uint32_t j = 0; j < pGradient->capacity(); j++) // add the gradient to the gradient table
            {
                pGradient->add(j, pGradientPart->at(j));
            }
        }
        
    }
    gradTable[pFocus->getId()] = pGradient;
}

std::vector<std::shared_ptr<Variable>> Graph::getVariableVec()
{
    return mVariableVec;
}

std::shared_ptr<Graph> GRAPH = std::make_shared<Graph>(); // the computational is a global graph


#endif // GRAPH_HPP