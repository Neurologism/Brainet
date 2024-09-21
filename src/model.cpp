//
// Created by servant-of-scietia on 20.09.24.
//

#include "model.hpp"

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


    const std::uint32_t trainingIterations = epochs * dataset.getTrainingSize() / batchSize;

    std::vector<std::shared_ptr<Variable>> graphInputs = dataset.getOutputs();
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    std::cout << std::setprecision(5) << std::fixed;

    double bestLoss = std::numeric_limits<double>::max();
    std::vector<std::vector<double>> bestParameters;
    std::uint32_t lastImprovement = 0;

    for(std::uint32_t iteration = 0; iteration < trainingIterations; iteration++)
    {
        // forward pass
        dataset.loadTrainingBatch(batchSize);
        GRAPH->forward(graphInputs);

        // log results
        const std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData();
        const std::shared_ptr<Tensor<double>> surrogateLoss = mLossVariables[1]->getData();

        std::cout << "{\n";
        std::cout << " \t \"iteration\": " << iteration << ",\n";
        std::cout << " \t \"loss\": " << loss->at(0) << ",\n";
        std::cout << " \t \"surrogate_loss\": " << surrogateLoss->at(0) << "\n";
        std::cout << "}\n";

        // backward pass
        GRAPH->backprop( mLearnableVariables, mGradientVariables, static_cast<double>(1)/batchSize); // backward pass

        // update weights
        std::visit([&](auto&& arg) {
            arg.update(mLearnableVariables); }, optimizer);
        continue;
        // validation pass
        dataset.loadValidationSet();
        GRAPH->forward(graphInputs);

        // log validation results
        std::shared_ptr<Tensor<double>> validationLoss = mLossVariables[0]->getData();
        std::shared_ptr<Tensor<double>> validationSurrogateLoss = mLossVariables[1]->getData();

        std::cout << " \t \"validation_loss\": " << validationLoss->at(0) << ",\n";
        std::cout << " \t \"validation_surrogate_loss\": " << validationSurrogateLoss->at(0) << "\n";
        std::cout << "}"<< std::endl;

        
        // early stopping
        double currentLoss = validationSurrogateLoss->at(0);

        if (currentLoss < bestLoss)
        {
            bestLoss = currentLoss;

            bestParameters.clear();
            for (const std::shared_ptr<Variable>& parameter : mLearnableVariables)
            {
                bestParameters.emplace_back();
                for (std::uint32_t i = 0; i < parameter->getData()->capacity(); i++)
                {
                    bestParameters.back().push_back(parameter->getData()->at(i));
                }
            }
            lastImprovement = iteration;
        }
        else if ( lastImprovement + earlyStoppingPatience <= iteration)
        {
            // std::cout << "Early stopping after " << iteration << " iterations.\t\t\t\t\t" << std::endl;
            // std::cout << "Best validation loss: " << bestLoss << std::endl;
            if (!bestParameters.empty())
            {
                for (std::uint32_t i = 0; i < mLearnableVariables.size(); i++)
                {
                    for (std::uint32_t j = 0; j < mLearnableVariables[i]->getData()->capacity(); j++)
                    {
                        mLearnableVariables[i]->getData()->set(j, bestParameters[i][j]);
                    }
                }
            }
            break;
        }
    }

    // std::cout<< "Training finished." << std::endl;

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