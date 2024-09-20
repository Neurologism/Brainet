//
// Created by servant-of-scietia on 20.09.24.
//

#include "variable.hpp"


std::uint32_t Variable::msCounter = 0;


Variable::Variable(const std::shared_ptr<Operation> &op, const std::vector<std::shared_ptr<Variable>> &parents, const std::vector<std::shared_ptr<Variable>> &children, const std::shared_ptr<Tensor<double>> &data)
{
    mId = msCounter++;
    mpOperation = op;
    mParents = parents;
    mChildren = children;
    mpDataTensor = data;
    if (op != nullptr)
    {
        mOperationName = op->getName();
    }
    else
    {
        mOperationName = "INPUT";
    }
};

std::shared_ptr<Operation> Variable::getOperation()
{
    return mpOperation;
}

void Variable::setOperation(const std::shared_ptr<Operation> &op)
{
    mpOperation = op;
}

std::vector<std::shared_ptr<Variable>> &Variable::getConsumers()
{
    return mChildren;
}

std::vector<std::shared_ptr<Variable>> &Variable::getInputs()
{
    return mParents;
}

std::shared_ptr<Tensor<double>> &Variable::getData()
{
    return mpDataTensor;
}

void Variable::setData(const std::shared_ptr<Tensor<double>> &data)
{
    mpDataTensor = data;
}

std::uint32_t Variable::getId() const
{
    return mId;
}

void Variable::connectVariables(const std::shared_ptr<Variable> &parent, const std::shared_ptr<Variable> &child)
{
    parent->getConsumers().push_back(child);
    child->getInputs().push_back(parent);
}

void Variable::disconnectVariables(const std::shared_ptr<Variable> &parent, const std::shared_ptr<Variable> &child)
{
    std::erase(parent->getConsumers(), child);
    std::erase(child->getInputs(), parent);
}