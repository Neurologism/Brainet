//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/operation.hpp"

void Operation::setVariable(std::shared_ptr<Variable> variable)
{
    mpVariable = variable;
}

std::shared_ptr<Variable> Operation::getVariable()
{
    if (mpVariable == nullptr)
    {
        throw std::runtime_error("variable is not set"); // this should never happen
    }
    return mpVariable;
}