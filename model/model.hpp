#ifndef MODEL_HPP
#define MODEL_HPP

#include "graph.hpp"
#include "module/module.hpp"
#include "operation/activation_function/activation_function.hpp"
#include "optimizers/optimizer.hpp"
#include "preprocessing/split.hpp"

class Model
{
protected:

    typedef std::vector<std::vector<double>> Vector2D;

    // everything needed for the graph
    std::vector<std::shared_ptr<Variable>> mLearnableVariables; // all variables that can be learned by the learning algorithm
    std::vector<std::shared_ptr<Variable>> mInputVariables;     // all variables that are used as input for the data
    std::vector<std::shared_ptr<Variable>> mTargetVariables;    // all variables that are used as input for the labels
    std::vector<std::shared_ptr<Variable>> mOutputVariables;    // all variables that are used as output 
    std::vector<std::shared_ptr<Variable>> mLossVariables;      // all variables that are used as output for the loss
    std::vector<std::shared_ptr<Variable>> mBackpropVariables;  // all variables that are used as starting point for the backpropagation and are leafs of the model subgraph

    virtual ~Model(){};

    /**
     * @brief 
     */
    void train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStoppingIteration = 20, double split = 0.8 );


    void test(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels);
};

void Model::train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStoppingIteration, double split)
{
    for (std::uint32_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].size() != inputs[0].size())
        {
            throw std::invalid_argument("Model::train: the size of all inputs and labels must be the same");
        }
    }
    for (std::uint32_t i = 1; i < labels.size(); i++)
    {
        if (labels[i].size() != labels[0].size())
        {
            throw std::invalid_argument("Model::train: the size of all inputs and labels must be the same");
        }
    }
    if ( inputs[0].size() != labels[0].size())
    {
        throw std::invalid_argument("Model::train: the size of all inputs and labels must be the same");
    }

    const std::uint32_t dataSamples = inputs[0].size();

    const std::uint32_t trainingIterations = epochs * dataSamples / batchSize;

    

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

        std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
        graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());
        GRAPH->forward(graphInputs); // forward pass


        std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData();
        
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