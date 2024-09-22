//
// Created by servant-of-scietia on 20.09.24.
//

#include "model.hpp"

#include "logger.hpp"

bool Model::earlyStopping(const std::uint32_t &epoch, std::uint32_t &bestEpoch, const std::uint32_t &earlyStoppingPatience, const double &error, double &bestError, std::vector<std::shared_ptr<Tensor<double>>> &bestParameters)
{
    if (error < bestError)
    {
        bestError = error;
        bestEpoch = epoch;

        bestParameters.clear();
        for (const std::shared_ptr<Variable>& parameter : mLearnableVariables)
        {
            bestParameters.push_back(std::make_shared<Tensor<double>>(*parameter->getData()));
        }
    }
    else if (bestEpoch + earlyStoppingPatience <= epoch)
    {
        for (const std::shared_ptr<Variable>& parameter : mLearnableVariables)
        {
            parameter->setData(bestParameters[parameter->getId()]);
        }
        return true;
    }
    return false;
}


std::shared_ptr<Module> Model::addModule(const ModuleVariant &module)
{
    const std::shared_ptr<Module> pModule = std::visit([]<typename T0>(T0&& arg) {
        // Assuming all types in the variant can be dynamically cast to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<T0>>(arg));}, ModuleVariant{module});

    mModules.push_back(pModule);
    mModuleMap[pModule->getName()] = pModule;

    auto learnableVariables = pModule->getLearnableVariables();
    mLearnableVariables.insert(mLearnableVariables.end(), learnableVariables.begin(), learnableVariables.end());
    auto gradientVariables = pModule->getGradientVariables();
    mGradientVariables.insert(mGradientVariables.end(), gradientVariables.begin(), gradientVariables.end());
    if (std::dynamic_pointer_cast<Loss>(pModule) != nullptr)
    {
        auto lossVariables = pModule->getOutputs();
        mLossVariables.insert(mLossVariables.end(), lossVariables.begin(), lossVariables.end());
    }

    return pModule;
}

void Model::addSequential(const std::vector<ModuleVariant> &modules)
{
    std::vector<std::shared_ptr<Module>> pModules;
    for (const ModuleVariant& module : modules)
    {
        pModules.push_back(addModule(module));
    }

    // connect Modules in sequential order
    for (std::uint32_t i = 0; i < pModules.size() - 1; i++)
    {
        connectModules(pModules[i], pModules[i+1]);
    }
}

void Model::connectModules(const std::shared_ptr<Module> &startModule, const std::shared_ptr<Module> &endModule)
{
    Variable::connectVariables(startModule->getOutputs()[0], endModule->getInputs()[0]);
    if (std::dynamic_pointer_cast<Loss>(endModule) != nullptr)
    {
        Variable::connectVariables(startModule->getOutputs()[0], endModule->getInputs()[1]);
    }
}

void Model::connectModules(const std::string &startModule, const std::string &endModule)
{
    connectModules(mModuleMap[startModule], mModuleMap[endModule]);
}

void Model::train(Dataset &dataset, const std::string& inputModule, const std::string& lossModule, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingPatience)
{
    Dropout::deactivateAveraging();

    Variable::connectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    Variable::connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    Variable::connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);

    std::vector<std::shared_ptr<Variable>> graphInputs = dataset.getOutputs();
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    double bestValidationSurrogateLoss = std::numeric_limits<double>::max();
    std::uint32_t bestEpoch = 0;

    std::vector<std::shared_ptr<Tensor<double>>> bestParameters;


    for (std::uint32_t epoch = 0; epoch < epochs; epoch++)
    {
        dataset.shuffleTrainingSet();


        while (dataset.goodBatch(batchSize))
        {
            dataset.loadTrainingBatch(batchSize);

            GRAPH->forward(graphInputs); // forward pass
            GRAPH->backprop( mLearnableVariables, mGradientVariables, static_cast<double>(1)/batchSize); // backward pass

            std::visit([&](auto&& arg) {
                arg.update(mLearnableVariables); }, optimizer); // update parameters

            // log and store results
            const double loss = mLossVariables[0]->getData()->at(0);
            const double surrogateLoss = mLossVariables[1]->getData()->at(0);

            Logger::logIteration(loss, surrogateLoss);
        }

        if (dataset.hasValidationSet())
        {
            dataset.loadValidationSet();
            GRAPH->forward(graphInputs); // validate

            // store results
            const double validationLoss = mLossVariables[0]->getData()->at(0);
            const double validationSurrogateLoss = mLossVariables[1]->getData()->at(0);

            Logger::logEpoch(validationLoss, validationSurrogateLoss);

            if (earlyStopping(epoch, bestEpoch, earlyStoppingPatience, validationSurrogateLoss, bestValidationSurrogateLoss, bestParameters))
            {
                break;
            }
        }
    }

    Variable::disconnectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    Variable::disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    Variable::disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);
}

void Model::test(Dataset &dataset, const std::string& inputModule, const std::string& lossModule)
{
    Dropout::activateAveraging();

    Variable::connectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    Variable::connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    Variable::connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);

    std::vector<std::shared_ptr<Variable>> graphInputs = dataset.getOutputs();
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    dataset.loadTestSet();
    GRAPH->forward(graphInputs); // forward pass

    const std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData();
    const std::shared_ptr<Tensor<double>> surrogateLoss = mLossVariables[1]->getData();

    std::cout << "{\n";
    std::cout << " \t \"test_loss\": " << loss->at(0) << ",\n";
    std::cout << " \t \"test_surrogate_loss\": " << surrogateLoss->at(0) << "\n";
    std::cout << "}"<< std::endl;

    Variable::disconnectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    Variable::disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    Variable::disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);
}