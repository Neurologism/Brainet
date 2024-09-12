#ifndef MODEL_HPP
#define MODEL_HPP

#include "../graph.hpp"
#include "../module/module.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../optimizer/optimizer.hpp"
#include "../preprocessing/split.hpp"

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
     * @param inputs the input data
     * @param labels the labels
     * @param epochs the number of epochs
     * @param batchSize the size of the batch
     * @param optimizer the optimizer to use
     * @param earlyStoppingIteration the number of iterations to wait for early stopping
     * @param split the split between training and validation data
     * @note the function will split the data into training and validation data per default
     */
    void train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, OptimizerVariant optimizer, std::uint32_t const earlyStoppingIteration = 20);

    /**
     * @brief function to test the model
     * @param inputs the input data
     * @param labels the labels
     * @note the function will print the error of the model
     */
    void test(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels);

    friend class Ensemble;
};

void Model::train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStoppingIteration, double split)
{

    Dropout::deactivateAveraging();

    const std::uint32_t dataSamples = inputs[0].size();

    const std::uint32_t trainingIterations = epochs * dataSamples / batchSize;

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    Vector2D trainData;
    Vector2D trainLabel;

    Vector2D validationData;
    Vector2D validationLabel;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    preprocessing::splitData(inputs[0], labels[0], split, trainData, validationData, trainLabel, validationLabel); // support multiple inputs and labels in the future 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::cout << std::setprecision(5) << std::fixed;

    double bestLoss = std::numeric_limits<double>::max();
    std::vector<std::vector<double>> bestParameters;
    std::uint32_t lastImprovement = 0;

    for(std::uint32_t iteration = 0; iteration < trainingIterations; iteration++)
    {
        Vector2D batchData;
        Vector2D batchLabel;

        // generate random batch
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, trainData.size() - 1);

        for (std::uint32_t i = 0; i < batchSize; i++)
        {
            std::uint32_t randomIndex = dis(gen);
            batchData.push_back(trainData[randomIndex]);
            batchLabel.push_back(trainLabel[randomIndex]);
        }

        mInputVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(batchData));
        mTargetVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(batchLabel)); // support multiple inputs and labels in the future

        
        GRAPH->forward(graphInputs); // forward pass


        std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData(); 
        std::shared_ptr<Tensor<double>> surrogateLoss = mLossVariables[1]->getData();
        
        std::cout << "Iteration: " << iteration << "\t Loss: " << loss->at(0) << "\t Surrogate loss: " << surrogateLoss->at(0);

        GRAPH->backprop( mLearnableVariables, mBackpropVariables); // backward pass

        for(std::uint32_t i = 0; i < mLearnableVariables.size(); i++)
        {
            std::shared_ptr<Tensor<double>> gradient = GRAPH->getGradient(mLearnableVariables[i]);
            for (std::uint32_t j = 0; j < mLearnableVariables[i]->getData()->capacity(); j++)
            {
                gradient->divide(j, batchSize);
            }
        }
        
        std::visit([&](auto&& arg) {
            arg.update(mLearnableVariables); }, optimizer); // update weights


        // validation
        mInputVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationData));
        mTargetVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationLabel)); // support multiple inputs and labels in the future

        GRAPH->forward(graphInputs); // forward pass

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

void Model::test(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels)
{
    for (std::uint32_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].size() != inputs[0].size())
        {
            throw std::invalid_argument("Model::test: the size of all inputs and labels must be the same");
        }
    }
    for (std::uint32_t i = 1; i < labels.size(); i++)
    {
        if (labels[i].size() != labels[0].size())
        {
            throw std::invalid_argument("Model::test: the size of all inputs and labels must be the same");
        }
    }
    if ( inputs[0].size() != labels[0].size())
    {
        throw std::invalid_argument("Model::test: the size of all inputs and labels must be the same");
    }

    Dropout::activateAveraging();

    const std::uint32_t dataSamples = inputs[0].size();

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());
    graphInputs.insert(graphInputs.end(), mLearnableVariables.begin(), mLearnableVariables.end());

    for (std::uint32_t i = 0; i < inputs.size(); i++)
    {
        mInputVariables[i]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(inputs[i]));
    }
    for (std::uint32_t i = 0; i < labels.size(); i++)
    {
        mTargetVariables[i]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(labels[i]));
    }

    GRAPH->forward(graphInputs); // forward pass

    std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData();
    std::shared_ptr<Tensor<double>> surrogateLoss = mLossVariables[1]->getData();

    std::cout << "Test loss: " << loss->at(0) << "\t Test surrogate loss: " << surrogateLoss->at(0) << std::endl;
}


// ModelVariant

#include "sequential.hpp"

using ModelVariant = std::variant<SequentialModel>;


#endif // MODEL_HPP