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
    std::vector<std::shared_ptr<Variable>> mOutputVariables;    // all variables that are used as output 
    std::vector<std::shared_ptr<Variable>> mLossVariables;      // all variables that are used as output for the loss
    std::vector<std::shared_ptr<Variable>> mBackpropVariables;  // all variables that are used as starting point for the backpropagation and are leafs of the model subgraph


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
    }

    static void connectModules(const std::shared_ptr<Module> &startModule, const std::shared_ptr<Module> &endModule)
    {
        startModule->addInput(endModule->getVariable(1), endModule->getUnits());
        endModule->addOutput(startModule->getVariable(0));
    }

    void connectModules(const std::string &startModule, const std::string &endModule)
    {
        connectModules(mModuleMap[startModule], mModuleMap[endModule]);
    }

    /**
     * @brief function to train the model
     * @param dataset the dataset to train the model
     * @param epochs the number of epochs
     * @param batchSize the size of the batch
     * @param optimizer the optimizer to use
     * @param earlyStoppingIteration the number of iterations to wait for early stopping
     */
    void train(const Dataset &dataset, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingIteration = 20);

    /**
     * @brief function to test the model
     * @note the function will print the error of the model
     */
    void test(const Dataset &dataset);

    friend class Ensemble;
};

inline void Model::train(const Dataset &dataset, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingIteration)
{

    Dropout::deactivateAveraging();

    const std::uint32_t trainingIterations = epochs * dataset.getTrainingSize() / batchSize;

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());
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
        
        std::cout << "Iteration: " << iteration << "\t Loss: " << loss->at(0) << "\t Surrogate loss: " << surrogateLoss->at(0);

        // backward pass
        GRAPH->backprop( mLearnableVariables, mBackpropVariables); // backward pass

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

        std::cout << "\t Validation loss: " << validationLoss->at(0) << "\t Validation surrogate loss: " << validationSurrogateLoss->at(0) << std::endl;

        
        // early stopping
        double currentLoss = validationSurrogateLoss->at(0);

        if (currentLoss < bestLoss)
        {
            bestLoss = currentLoss;
            
            bestParameters.clear();
            for (std::shared_ptr<Variable> parameter : mLearnableVariables)
            {
                bestParameters.push_back({});
                for (std::uint32_t i = 0; i < parameter->getData()->capacity(); i++)
                {
                    bestParameters.back().push_back(parameter->getData()->at(i));
                }
            }
            lastImprovement = iteration;
        }
        else if ( lastImprovement + earlyStoppingIteration <= iteration)
        {
            std::cout << "Early stopping after " << iteration << " iterations.\t\t\t\t\t" << std::endl;
            std::cout << "Best validation loss: " << bestLoss << std::endl;
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

    std::cout<< "Training finished." << std::endl;
}

inline void Model::test(const Dataset &dataset)
{
    Dropout::activateAveraging();

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    dataset.loadTestSet();
    GRAPH->forward(graphInputs); // forward pass

    std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData();
    std::shared_ptr<Tensor<double>> surrogateLoss = mLossVariables[1]->getData();

    std::cout << "Test loss: " << loss->at(0) << "\t Test surrogate loss: " << surrogateLoss->at(0) << std::endl;
}


// ModelVariant

#include "sequential.hpp"

using ModelVariant = std::variant<SequentialModel>;


#endif // MODEL_HPP