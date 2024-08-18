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
    typedef std::shared_ptr<Variable> VariablePtr;
    typedef std::map<VariablePtr, std::shared_ptr<Tensor<double>>> GradTable;

    std::vector<VariablePtr> mVariableVec; // all variables in the graph
    GradTable mGradTable; // the gradient table for the variables
    
    /**
     * @brief This function builds the gradient table for the variable focus. It is a recursive function that calculates the gradient of the focus variable with respect to all other variables in the graph.
     * To do this it uses dynamic programming.
     * @param focus The variable for which the gradient is calculated.
     * @param gradTable The gradient table that stores already calculated gradients.
     */
    void mBuildGrad(VariablePtr pFocus, GradTable & pGradTable);
    /**
     * @brief This function performs a topological sort on the graph and returns the sorted variables.
     * @return std::vector<VariablePtr> The sorted variables.
     */
    std::vector<VariablePtr> mTopoSort( std::vector<VariablePtr> & inputVariables );

public:
    Graph() = default;
    ~Graph() = default;

    /**
     * @brief This function simply executes the operations of the graph in topological order.
     * @param inputVariables The Variables from which the data is propagated through the graph.
     */
    void forward(std::vector<VariablePtr> & inputVariables);

    /**
     * @brief This function calculates the gradients of the target variables with respect to the variables in the differentiate vector.
     * It utilizes the well know general backpropagation algorithm and is implemented in a dynamic programming fashion.
     * @param targetVariables The variables for which the gradients are calculated.
     * @param outputVariables The target variables are computed with respect to the output variables.
     * @return The gradients of the target variables in the same order as the target variables.
     */
    void backprop(std::vector<VariablePtr> & targetVariables, std::vector<VariablePtr> & outputVariables);

    /**
     * @brief This function returns all variables in the graph.
     * @return std::vector<VariablePtr> The variables in the graph.
     */
    std::vector<VariablePtr> getVariableVec();

    /**
     * @brief This function returns the gradient of a variable.
     * @param pVar The variable for which the gradient is calculated.
     * @return The gradient of the variable.
     */
    std::shared_ptr<Tensor<double>> getGradient(VariablePtr pVar);

    /**
     * @brief This function adds a variable to the graph.
     * @param var The variable to be added.
     * @return VariablePtr The added variable.
     */
    VariablePtr addVariable(const VariablePtr & pVar);

    /**
     * @brief This function removes a variable from the graph.
     * @param var The variable to be removed.
     */
    void removeVariable(const VariablePtr & pVar);
}; 


std::vector<std::shared_ptr<Variable>> Graph::mTopoSort( std::vector<VariablePtr> & inputVariables )
{
    std::vector<VariablePtr> pSorted;
    std::vector<bool> visited(mVariableVec.size(), false);
    // depth first search
    std::function<void(VariablePtr)> dfs = [&](VariablePtr pVar) 
    {
        if(visited[pVar->getId()]) return;
        visited[pVar->getId()] = true;
        for (VariablePtr pChild : pVar->getConsumers())
        {
            if (!visited[pChild->getId()])
            {
                dfs(pChild);
            }
        }
        pSorted.push_back(pVar);
    };

    for (VariablePtr pVar : inputVariables) // call for all variables
    {
        if (!visited[pVar->getId()])
        {
            dfs(pVar);
        }
    }
    std::reverse(pSorted.begin(), pSorted.end()); // reverse the order to get the topological order

    std::vector<VariablePtr> selectedVariables;
    for (VariablePtr pVar : pSorted) // remove all variables that are not descendats of only input variables
    {
        bool isDescendant = true;
        for (VariablePtr pInput : pVar->getInputs())
        {
            if ( visited[pInput->getId()] == false )
            {
                isDescendant = false;
                visited[pVar->getId()] = false;
                break;
            }
        }

        if (isDescendant)
        {
            selectedVariables.push_back(pVar);
        }
    }
    return selectedVariables;
}

void Graph::forward(std::vector<VariablePtr> & inputVariables)
{
    std::vector<VariablePtr> pSorted = mTopoSort( inputVariables ); // sort the variables topologically and get only the variables that are descendants of the input variables
    for (VariablePtr var : pSorted)
    {
        if (var->getOperation() != nullptr) // if the variable has an operation, execute it
        {
            var->getOperation()->f(var->getInputs()); // execute the operation
        }
    }
}

void Graph::backprop(std::vector<VariablePtr> & targetVariables, std::vector<VariablePtr> & outputVariables)
{
    mGradTable.clear(); // clear the gradient table
    GradTable gradTable; // the gradient table for the variables

    for(VariablePtr pVar : outputVariables) // initialize the gradient table with the output variables
    {
        gradTable[pVar] = std::make_shared<Tensor<double>>(Tensor<double>(pVar->getData()->shape())); // initialize the gradient table with the output variables
        for(std::uint32_t i = 0; i < pVar->getData()->capacity(); i++)
        {
            gradTable[pVar]->set(i, 1); // output variables have a gradient of 1 and are considered as leaf nodes
        }
    }

    for (VariablePtr pVar : targetVariables) // calculate the gradients for all target variables
    {
        mBuildGrad(pVar, gradTable); // build the gradient table for the target variables
        mGradTable[pVar] = gradTable[pVar]; // store the gradients in the global gradient table
    }
}

void Graph::mBuildGrad(VariablePtr pFocus, GradTable & gradTable)
{
    // error handling
    if (gradTable.find(pFocus) != gradTable.end()) // if the gradient is already calculated, return
    {
        return;
    }
    if (pFocus->getConsumers().empty())
    {
        return; // empty consumers -> leaf node (should handle no gradient in backprop)
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
        VariablePtr pConsumer = pFocus->getConsumers().at(i);
        std::shared_ptr<Operation> pOperation = pConsumer->getOperation();
        mBuildGrad(pConsumer, gradTable); // build the gradient table for the consumer
        std::shared_ptr<Tensor<double>> pGradientPart = pOperation->bprop(pConsumer->getInputs(), pFocus, gradTable[pConsumer]); // calculate the gradient of the consumer with respect to the focus variable
        if (pGradientPart->shape() != pFocus->getData()->shape())
        {
            throw std::runtime_error("Gradient shape does not match variable shape");
        }
        if(pGradient == nullptr)
        {
            pGradient = pGradientPart;
        }
        else
        {
            for (std::uint32_t j = 0; j < pGradient->capacity(); j++) // add the gradient to the gradient table
            {
                pGradient->add(j, pGradientPart->at(j));
            }
        }
        
    }
    gradTable[pFocus] = pGradient;
}

std::vector<std::shared_ptr<Variable>> Graph::getVariableVec()
{
    return mVariableVec;
}

std::shared_ptr<Tensor<double>> Graph::getGradient(VariablePtr pVar)
{
    if(mGradTable.find(pVar) == mGradTable.end()) throw std::runtime_error("Variable not in gradient table");
    return mGradTable[pVar];
}

std::shared_ptr<Variable> Graph::addVariable(const VariablePtr & pVar)
{
    mVariableVec.push_back(pVar); 
    if(pVar->getOperation()!=nullptr)pVar->getOperation()->setVariable(mVariableVec.back()); // address of variable has changed -> invalidation of pointers; not nice but works
    return mVariableVec.back();
}

void Graph::removeVariable(const VariablePtr & pVar)
{
    mVariableVec.erase(std::find(mVariableVec.begin(), mVariableVec.end(), pVar));
}

std::shared_ptr<Graph> GRAPH = std::make_shared<Graph>(); // global graph object


#endif // GRAPH_HPP