#ifndef MODEL_HPP
#define MODEL_HPP

#include "../graph.hpp"
#include "../module/module.hpp"
#include "../operation/activation_function/activation_function.hpp"
#include "../optimizers/optimizer.hpp"
#include "../preprocessing/split.hpp"

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

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());

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


        std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData(); // support multiple loss variables in the future
        
        std::cout << "Batch: " << iteration << " Training-error: " << loss->at(0);

        GRAPH->backprop( mLearnableVariables, mBackpropVariables); // backward pass
        
        std::visit([&](auto&& arg) {
            arg.update(mLearnableVariables, batchSize); }, optimizer); // update weights


        // validation
        mInputVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationData));
        mTargetVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationLabel)); // support multiple inputs and labels in the future

        GRAPH->forward(graphInputs); // forward pass

        std::shared_ptr<Tensor<double>> validationLoss = mLossVariables[0]->getData();

        std::cout << " Validation-error: " << validationLoss->at(0) << std::endl;

        
        // early stopping
        double currentLoss = validationLoss->at(0);

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
            std::cout << "Early stopping after " << iteration << " iterations." << std::endl;
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

    const std::uint32_t dataSamples = inputs[0].size();

    std::vector<std::shared_ptr<Variable>> graphInputs = mInputVariables;
    graphInputs.insert(graphInputs.end(), mTargetVariables.begin(), mTargetVariables.end());

    mInputVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(inputs[0]));
    mTargetVariables[0]->getData() = std::make_shared<Tensor<double>>(Matrix<double>(labels[0])); // support multiple inputs and labels in the future

    GRAPH->forward(graphInputs); // forward pass

    std::shared_ptr<Tensor<double>> loss = mLossVariables[0]->getData(); // support multiple loss variables in the future
    
    std::cout << "Test-error: " << loss->at(0) << std::endl;
}

    

#endif // MODEL_HPP