//
// Created by servant-of-scietia on 20.09.24.
//

#include "graph.hpp"

std::vector<std::shared_ptr<Variable>> Graph::mTopologicalSort( std::vector<VariablePtr> & inputVariables ) const
{
    std::vector<VariablePtr> pSorted;
    std::vector<bool> visited(mVariableVec.size(), false);
    // depth-first search
    std::function<void(VariablePtr)> dfs = [&](const VariablePtr& pVar)
    {
        if(visited[pVar->getId()]) return;
        visited[pVar->getId()] = true;
        for (const VariablePtr& pChild : pVar->getConsumers())
        {
            if (!visited[pChild->getId()])
            {
                dfs(pChild);
            }
        }
        pSorted.push_back(pVar);
    };

    for (const VariablePtr& pVar : inputVariables) // call for all variables
    {
        if (!visited[pVar->getId()])
        {
            dfs(pVar);
        }
    }
    std::ranges::reverse(pSorted); // reverse the order to get the topological order

    std::vector<VariablePtr> selectedVariables;
    for (const VariablePtr& pVar : pSorted) // remove all variables that are not descendants of only input variables
    {
        bool isDescendant = true;
        for (const VariablePtr& pInput : pVar->getInputs())
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

void Graph::forward(std::vector<VariablePtr> & inputVariables) const
{
    for (std::vector<VariablePtr> pSorted = mTopologicalSort( inputVariables ); const VariablePtr& var : pSorted)
    {
        if (var->getOperation() != nullptr) // if the variable has an operation, execute it
        {
            var->getOperation()->f(var->getInputs()); // execute the operation
        }
    }
}

void Graph::backprop(std::vector<VariablePtr> & targetVariables, std::vector<VariablePtr> & leafVariables, double leafInitValue)
{
    mGradTable.clear(); // clear the gradient table
    GradTable gradTable; // the gradient table for the variables

    for(const VariablePtr& pVar : leafVariables) // initialize the gradient table
    {
        gradTable[pVar] = std::make_shared<Tensor>(Tensor(pVar->getData()->shape())); // set leafs to leafInitValue
        for(std::uint32_t i = 0; i < pVar->getData()->capacity(); i++)
        {
            gradTable[pVar]->set(i, leafInitValue);
        }
    }

    for (const VariablePtr& pVar : targetVariables) // calculate the gradients for all target variables
    {
        mBuildGrad(pVar, gradTable); // build the gradient table for the target variables
        mGradTable[pVar] = gradTable[pVar]; // store the gradients in the global gradient table
    }
}

void Graph::mBuildGrad(VariablePtr pFocus, GradTable & gradTable) // NOLINT
{
    // error handling
    if (gradTable.contains(pFocus)) // if the gradient is already calculated, return
    {
        return;
    }
    if (pFocus->getConsumers().empty())
    {
        return; // empty consumers → leaf node (should handle no gradient in backprop)
    }
    if (pFocus->getConsumers().empty())
    {
        throw std::runtime_error("Variable has no consumers");
    }
    if (pFocus->getData()->dimensionality() == 0)
    {
        throw std::runtime_error("Variable has no data");
    }

    std::shared_ptr<Tensor> pGradient;
    for (std::uint32_t i = 0; i < pFocus->getConsumers().size(); i++) // the sum the consumer gradients is the gradient of the variable
    {
        // load stuff
        VariablePtr pConsumer = pFocus->getConsumers().at(i);
        std::shared_ptr<Operation> pOperation = pConsumer->getOperation();
        mBuildGrad(pConsumer, gradTable); // build the gradient table for the consumer
        std::shared_ptr<Tensor> pGradientPart = pOperation->bprop(pConsumer->getInputs(), pFocus, gradTable[pConsumer]); // calculate the gradient of the consumer with respect to the focus variable
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

std::shared_ptr<Tensor> Graph::getGradient(const VariablePtr& pVar)
{
    if(!mGradTable.contains(pVar)) throw std::runtime_error("Variable not in gradient table");
    return mGradTable[pVar];
}

std::shared_ptr<Variable> Graph::addVariable(const VariablePtr & pVar)
{
    mVariableVec.push_back(pVar);
    if(pVar->getOperation()!=nullptr)pVar->getOperation()->setVariable(mVariableVec.back()); // address of variable has changed →
    // invalidation of pointers;
    // not nice but works
    return mVariableVec.back();
}

void Graph::removeVariable(const VariablePtr & pVar)
{
    mVariableVec.erase(std::ranges::find(mVariableVec, pVar));
}


