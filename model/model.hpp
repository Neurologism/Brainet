#ifndef MODEL_HPP
#define MODEL_HPP

#include "graph.hpp"
#include "module/module.hpp"
#include "operation/activation_function/activation_function.hpp"
#include "optimizers/optimizer.hpp"


class Model
{
protected:

    typedef std::vector<std::vector<double>> Vector2D;

    // everything needed for the graph
    std::vector<std::shared_ptr<Variable>> mLearnableVariables; // all variables that can be learned by the learning algorithm
    std::vector<std::shared_ptr<Variable>> mInputVariables;     // all variables that are used as input for the data
    std::vector<std::shared_ptr<Variable>> mTargetVariables;     // all variables that are used as input for the labels
    std::vector<std::shared_ptr<Variable>> mOutputVariables;    // all variables that are used as output 
    std::vector<std::shared_ptr<Variable>> mLossVariables;      // all variables that are used as output for the loss
    std::vector<std::shared_ptr<Variable>> mBackpropVariables;  // all variables that are used as starting point for the backpropagation and are leafs of the model subgraph

    virtual ~Model(){};

    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param dataPairs Vector giving sets of {trainData, trainLabel, validationData, validationLabel}. Each set index needs to have input and output points in mDataPairs.
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param optimizer The optimizer to use.
     * @param earlyStopping The number of epochs without improvement after which to stop training. If 0, no early stopping is used.
     */
    void train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, double split = 0.8, std::uint32_t const earlyStopping = 0);

    /**
     * @brief Shortcut for training a model without validation data.
     * @param data The data.
     * @param label The label.
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param learning_rate The learning rate.
     */
    void train(Vector2D const data, Vector2D const label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer);

    /**
     * @brief This function trains the model with the given data and label. It uses the backpropagation algorithm to update the learnable parameters.
     * @param trainData The training data.
     * @param trainLabel The training label.
     * @param validationData The validation data.
     * @param validationLabel The validation label.
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param earlyStopping The number of epochs without improvement after which to stop training. If 0, no early stopping is used.
     */
    void train(Vector2D const trainData, Vector2D const trainLabel, Vector2D const validationData, Vector2D const validationLabel, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t earlyStopping = 0);

    /**
     * @brief This function gets the test error of the model.
     * @param dataPairs Map distributing the data/label pairs to the input/output nodes according to their id. id : (data, label)
     */

    void test(std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> const & dataPairs);
    /**
     * @brief Shortcut for testing a model with only one data/label pair. Assumes the id is 0.
     * @param data The data.
     * @param label The label.
     */
    void test(Vector2D const data, Vector2D const label);
};

void Model::train(std::vector<std::array<Vector2D, 4>> const & dataPairs, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStopping)
{
    std::uint32_t const pairId = 0;

    if ( mDataPairs.find(pairId) == mDataPairs.end() )
    {
        throw std::runtime_error("no access point for data/label pair with id 0 found");
    }

    const std::uint32_t trainingIterations = epochs * dataPairs[pairId][0].size() / batchSize;

    Vector2D trainData = dataPairs[pairId][0];
    Vector2D trainLabel = dataPairs[pairId][1];

    Vector2D validationData;
    Vector2D validationLabel;

    std::cout << std::setprecision(5) << std::fixed;

    if(dataPairs[pairId][2].size())
    {
        validationData = dataPairs[pairId][2];
        validationLabel = dataPairs[pairId][3];
    }
    else std::cout << "No validation data found. Training without validation." << std::endl;

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

        mDataPairs[0].first->getData() = std::make_shared<Tensor<double>>(Matrix<double>(batchData));
        mDataPairs[0].second->getData() = std::make_shared<Tensor<double>>(Matrix<double>(batchLabel));

        GRAPH->forward();
        std::shared_ptr<Tensor<double>> loss = GRAPH->getOutput(mLossIndex);
        
        std::cout << "Batch: " << iteration << " Training-error: " << loss->at(0);

        std::vector<std::shared_ptr<Tensor<double>>> gradientTable = GRAPH->backprop(Module::getLearnableParameters()); // backpropagation
        
        std::visit([gradientTable, batchSize](auto&& arg) {
            arg.update(gradientTable, batchSize);}, optimizer);

        if ( dataPairs[pairId][2].size() )
        {
            mDataPairs[pairId].first->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationData));
            mDataPairs[pairId].second->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationLabel));

            GRAPH->forward();
            std::shared_ptr<Tensor<double>> validationLoss = GRAPH->getOutput(mLossIndex);
            std::cout << " Validation-error: " << validationLoss->at(0) << std::endl;

            if (earlyStopping)
            {
                double currentLoss = validationLoss->at(0);

                if (currentLoss < bestLoss)
                {
                    bestLoss = currentLoss;
                    
                    bestParameters.clear();
                    for (std::shared_ptr<Variable> parameter : Module::getLearnableParameters())
                    {
                        bestParameters.push_back({});
                        for (std::uint32_t i = 0; i < parameter->getData()->capacity(); i++)
                        {
                            bestParameters.back().push_back(parameter->getData()->at(i));
                        }
                    }
                    lastImprovement = iteration;
                }
                else if ( lastImprovement + earlyStopping <= iteration)
                {
                    std::cout << "Early stopping after " << iteration << " iterations." << std::endl;
                    std::cout << "Best validation loss: " << bestLoss << std::endl;
                    for (std::uint32_t i = 0; i < Module::getLearnableParameters().size(); i++)
                    {
                        for (std::uint32_t j = 0; j < Module::getLearnableParameters()[i]->getData()->capacity(); j++)
                        {
                            Module::getLearnableParameters()[i]->getData()->set(j, bestParameters[i][j]);
                        }
                    }
                    break;
                }
            }
        }
        else std::cout << std::endl;
    }

    std::cout<< "Training finished." << std::endl;
}

void Model::train(Vector2D const trainData, Vector2D const trainLabel, Vector2D const validationData, Vector2D const validationLabel, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t earlyStopping)
{
    std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> dataPairs;
    if (mDataPairs.find(0) == mDataPairs.end())
    {
        throw std::runtime_error("Assumed id 0, but no data/label pair with id 0 found");
    }
    dataPairs[0] = std::make_pair(trainData, trainLabel);
    dataPairs[1] = std::make_pair(validationData, validationLabel);
    train(dataPairs, epochs, batchSize, optimizer, earlyStopping);
}
        

void Model::test(std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> const & dataPairs)
{
    for (auto const & dataPair : dataPairs)
    {
        mDataPairs[dataPair.first].first->getData() = std::make_shared<Tensor<double>>(Matrix<double>(dataPair.second.first));
        mDataPairs[dataPair.first].second->getData() = std::make_shared<Tensor<double>>(Matrix<double>(dataPair.second.second));

        GRAPH->forward();
        std::shared_ptr<Tensor<double>> loss = GRAPH->getOutput(mLossIndex);
        std::cout << "Test Loss: " << loss->at(0) << std::endl; // print loss
    }
}


void Model::test(Vector2D const testData, Vector2D const testLabel)
{
    std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> dataPairs;
    if (mDataPairs.find(0) == mDataPairs.end())
    {
        throw std::runtime_error("Assumed id 0, but no data/label pair with id 0 found");
    }
    dataPairs[0] = std::make_pair(testData, testLabel);
    test(dataPairs);
}

#endif // MODEL_HPP