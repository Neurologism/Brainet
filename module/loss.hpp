#ifndef LOSS_HPP
#define LOSS_HPP

#include "../dependencies.hpp"
#include "../operation/loss_functions/loss_function.hpp"
#include "../operation/surrogate_loss_functions/surrogate_loss_function.hpp"

/**
 * @brief the loss module is intended for calculating the loss as well as the surrogate loss of the model.
 
 */
class Loss final : public Module
{
    std::shared_ptr<Variable> mTargetVariable; // storing the target
    std::shared_ptr<Variable> mLossVariable; // storing the loss
    std::shared_ptr<Variable> mSurrogateLossVariable; // storing the surrogate loss

    void createVariables(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction);

public:
    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @param name the name of the loss module
     * @note the surrogate loss function will be determined automatically
     */
    explicit Loss(const LossFunctionVariant& lossFunction, const std::string & name = "");

    /**
     * @brief add a loss function to the graph
     * @param lossFunction the operation representing the loss function.
     * @param surrogateLossFunction the operation representing the surrogate loss function.
     * @param name the name of the loss module
     */
    Loss(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction, const std::string & name = "");

    // void remove();

    /**
     * @brief used to initialize the module with the input and output variables.
     * @param initialInputs the input variables
     * @param initialOutputs the output variables
     */
    void __init__( std::vector<std::shared_ptr<Variable>> initialInputs, std::vector<std::shared_ptr<Variable>> initialOutputs ) override;

    /**
     * @brief function to get access to specific variables of the module.
     * @param index the index of the variable
     * @return the variable specified by the index
     * @note 0: surrogate loss variable
     * @note 1: loss variable
     * @note 2: target variable
     */
    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;
    
};

inline Loss::Loss(const LossFunctionVariant& lossFunction, const std::string & name) : Module(name)
{
    if (std::holds_alternative<ErrorRate>(lossFunction)) // Error Rate & Cross Entropy
    {
        createVariables(lossFunction, CrossEntropy());
    }
    else
    {
        throw std::invalid_argument("Loss::Loss: the loss function has no default surrogate loss function");
    }
}

inline Loss::Loss(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction, const std::string & name) : Module(name)
{
    createVariables(lossFunction, surrogateLossFunction);
}

inline void Loss::createVariables(const LossFunctionVariant &lossFunction, const SurrogateLossFunctionVariant &surrogateLossFunction)
{
    // add variables to the graph
    mTargetVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));

    mLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, LossFunctionVariant{lossFunction}), {mTargetVariable}, {})));

    mSurrogateLossVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(std::visit([]<typename T0>(T0&& arg) {
        return std::shared_ptr<Operation>(std::make_shared<std::decay_t<T0>>(arg));}, SurrogateLossFunctionVariant{surrogateLossFunction}), {mTargetVariable}, {})));
    
    // connections within the module
    mTargetVariable->getConsumers().push_back(mLossVariable);
    mTargetVariable->getConsumers().push_back(mSurrogateLossVariable);
}

// void Loss::remove()
// {
//     // delete other connections
//     for(auto input : mCostVariable->getInputs())
//     {
//         input->getConsumers().erase(std::find(input->getConsumers().begin(), input->getConsumers().end(), mCostVariable));
//     }

//     // delete the variables
//     GRAPH->removeVariable(mTargetVariable);

//     GRAPH->removeVariable(mCostVariable);
// }


void Loss::__init__(const std::vector<std::shared_ptr<Variable>> initialInputs, const std::vector<std::shared_ptr<Variable>> initialOutputs )
{
    if (initialInputs.size() != 1)
    {
        throw std::invalid_argument("Loss::__init__: the number of input variables must be 1");
    }
    if (!initialOutputs.empty())
    {
        throw std::invalid_argument("Loss::__init__: the number of output variables must be 0");
    }

    mLossVariable->getInputs().push_back(initialInputs[0]);
    mSurrogateLossVariable->getInputs().push_back(initialInputs[0]);
}

inline std::shared_ptr<Variable> Loss::getVariable(const std::uint32_t index)
{
    switch (index)
    {
    case 0:
        return mSurrogateLossVariable;
    case 1:
        return mLossVariable;
    case 2:
        return mTargetVariable;
    default:
        throw std::invalid_argument("Loss::getVariable: index out of range");
    }
}



#endif // LOSS_HPP