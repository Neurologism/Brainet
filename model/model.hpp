#ifndef MODEL_HPP
#define MODEL_HPP

#include "../graph.hpp"
#include "../module/module.hpp"
#include "../optimizer/optimizer.hpp"
#include "../module/dataset.hpp"

/**
 * @brief The Model class is intended to be used as a base class for all models.
 * @details The Model class provides the basic functionality to train and test a model.
 * It is intended to be extended by different model classes like SequentialModel or Ensemble that 
 * provide simple interfaces to create and train models and offer a constructor to create the model.
 * @note The Model class is abstract and cannot be instantiated.
 */
class Model
{

protected:
    // everything needed for the graph
    std::vector<std::shared_ptr<Variable>> mLearnableVariables; // all variables that can be learned by the learning algorithm
    std::vector<std::shared_ptr<Variable>> mGradientVariables;  // all variables that are used as starting point for the backpropagation and are leafs of the model subgraph
    std::vector<std::shared_ptr<Variable>> mLossVariables;      // all variables that are used as output for the loss

    std::vector<std::shared_ptr<Module>> mModules; // all modules of the model
    std::map<std::string, std::shared_ptr<Module>> mModuleMap; // map to access modules by name


public:

    void addModule(const ModuleVariant &module)
    {
        const std::shared_ptr<Module> modulePtr = std::visit([]<typename T0>(T0&& arg) {
            // Assuming all types in the variant can be dynamically cast to OPERATION*
            return std::shared_ptr<Module>(std::make_shared<std::decay_t<T0>>(arg));}, ModuleVariant{module});

        mModules.push_back(modulePtr);
        mModuleMap[modulePtr->getName()] = modulePtr;

        auto learnableVariables = modulePtr->getLearnableVariables();
        mLearnableVariables.insert(mLearnableVariables.end(), learnableVariables.begin(), learnableVariables.end());
        auto gradientVariables = modulePtr->getGradientVariables();
        mGradientVariables.insert(mGradientVariables.end(), gradientVariables.begin(), gradientVariables.end());
        if (std::dynamic_pointer_cast<Loss>(modulePtr) != nullptr)
        {
            auto lossVariables = modulePtr->getOutputs();
            mLossVariables.insert(mLossVariables.end(), lossVariables.begin(), lossVariables.end());
        }
    }

    static void connectModules(const std::shared_ptr<Module> &startModule, const std::shared_ptr<Module> &endModule)
    {
        connectVariables(startModule->getOutputs()[0], endModule->getInputs()[0]);
        if (std::dynamic_pointer_cast<Loss>(endModule) != nullptr)
        {
            connectVariables(startModule->getOutputs()[0], endModule->getInputs()[1]);
        }
    }

    void connectModules(const std::string &startModule, const std::string &endModule)
    {
        connectModules(mModuleMap[startModule], mModuleMap[endModule]);
    }

    /**
     * @brief function to train the model
     * @param dataset the dataset to train the model
     * @param inputModule the name of the input module
     * @param lossModule the name of the loss module
     * @param epochs the number of epochs
     * @param batchSize the size of the batch
     * @param optimizer the optimizer to use
     * @param earlyStoppingIteration the number of iterations to wait for early stopping
     */
    void train(Dataset &dataset, const std::string& inputModule, const std::string& lossModule, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingIteration);

    /**
     * @brief function to test the model
     * @note the function will print the error of the model
     */
    void test(Dataset &dataset, const std::string& inputModule, const std::string& lossModule);

    friend class Ensemble;
};

inline void Model::train(Dataset &dataset, const std::string& inputModule, const std::string& lossModule, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingIteration)
{
    Dropout::deactivateAveraging();

    connectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);


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

        // backward pass
        GRAPH->backprop( mLearnableVariables, mGradientVariables); // backward pass

        // normalize the gradient
        for(const auto & mLearnableVariable : mLearnableVariables)
        {
            const std::shared_ptr<Tensor<double>> gradient = GRAPH->getGradient(mLearnableVariable);
            for (std::uint32_t j = 0; j < mLearnableVariable->getData()->capacity(); j++)
            {
                gradient->divide(j, batchSize);
            }
        }

        // update weights
        std::visit([&](auto&& arg) {
            arg.update(mLearnableVariables); }, optimizer);

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
        else if ( lastImprovement + earlyStoppingIteration <= iteration)
        {
            // std::cout << "Early stopping after " << iteration << " iterations.\t\t\t\t\t" << std::endl;
            // std::cout << "Best validation loss: " << bestLoss << std::endl;
            for (std::uint32_t i = 0; i < mLearnableVariables.size(); i++)
            {
                for (std::uint32_t j = 0; j < mLearnableVariables[i]->getData()->capacity(); j++)
                {
                    mLearnableVariables[i]->getData()->set(j, bestParameters[i][j]);
                }
            }
            break;
        }
    }

    // std::cout<< "Training finished." << std::endl;

    disconnectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);
}

inline void Model::test(Dataset &dataset, const std::string& inputModule, const std::string& lossModule)
{
    Dropout::activateAveraging();

    connectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    connectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);

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

    disconnectVariables(dataset.getOutputs()[0], mModuleMap[inputModule]->getInputs()[0]);
    disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[0]);
    disconnectVariables(dataset.getOutputs()[1], mModuleMap[lossModule]->getInputs()[1]);
}


// ModelVariant

// #include "sequential.hpp"
//
// using ModelVariant = std::variant<SequentialModel>;


#endif // MODEL_HPP