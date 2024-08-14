#ifndef MODEL_HPP
#define MODEL_HPP

#include "graph.hpp"
#include "module/module.hpp"
#include "operation/activation_function/activation_function.hpp"
#include "optimizers/optimizer.hpp"

/**
 * @brief The model class is the main interface for the user to create a neural network. It is used to build the network and to train it.
 * The model class combines the graph, the module and the operation classes to create a neural network.
 */
class Model
{
    typedef std::vector<std::vector<double>> Vector2D;

    std::uint32_t mLossIndex; // the index of the loss function in the graph


    std::shared_ptr<Graph> mpGraph = std::make_shared<Graph>(); // the computational graph
    std::map<std::uint32_t, std::pair<std::shared_ptr<Variable>, std::shared_ptr<Variable>>> mDataPairs; // the data/label pairs


public:
    /**
     * @brief Construct a new Model object.
     */
    Model()
    {
        Module::setGraph(mpGraph); // set the static graph pointer in the module class
    };
    ~Model(){};
    /**
     * @brief This function loads the graph of the model into the module class.
     */
    void load();
    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param id The id of the data/label pair.
     */
    void sequential(std::vector<ModuleVariant> layers, std::uint32_t id = 0);

    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param norm The norm to use for regularization.
     * @param id The id of the data/label pair.
     */
    void sequential(std::vector<ModuleVariant> layers, NormVariant norm, std::uint32_t id = 0);

private:
    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param dataPairs Vector giving sets of {trainData, trainLabel, validationData, validationLabel}. Each set index needs to have input and output points in mDataPairs.
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param optimizer The optimizer to use.
     * @param earlyStopping The number of epochs without improvement after which to stop training. If 0, no early stopping is used.
     */
    void train(std::vector<std::array<Vector2D, 4>> const & dataPairs, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t earlyStopping);

public:
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

void Model::load()
{
    Module::setGraph(mpGraph);
}

void Model::sequential(std::vector<ModuleVariant> layers, std::uint32_t id)
{
    std::vector<std::shared_ptr<Module>> modules;

    for (ModuleVariant& layer : layers) {
        std::shared_ptr<Module> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ModuleVariant{layer});
        modules.push_back(cluster_ptr);
    }
    
    for(std::uint32_t i = 0; i < layers.size() - 1; i++)
    {
        modules[i]->addOutput(modules[i+1]->input());
        modules[i+1]->addInput(modules[i]->output(),modules[i]->getUnits());
    }

    mLossIndex = mpGraph->addOutput(modules.back()->output());
    if (mDataPairs.find(id) != mDataPairs.end())
    {
        throw std::runtime_error("id already exists");
    }
    // add error checks in the future
    std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(modules.front());
    std::shared_ptr<Cost> output = std::dynamic_pointer_cast<Cost>(modules.back());
    mDataPairs[id] = std::make_pair(input->data(), output->target());
}

void Model::sequential(std::vector<ModuleVariant> layers, NormVariant norm, std::uint32_t id)
{
    Dense::setDefaultNorm(norm);
    sequential(layers, id);
}


void Model::train(std::vector<std::array<Vector2D, 4>> const & dataPairs, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStopping)
{
    std::uint32_t const pairId = 0;

    if ( mDataPairs.find(pairId) == mDataPairs.end() )
    {
        throw std::runtime_error("no access point for data/label pair with id 0 found");
    }

    const std::uint32_t trainingIterations = epochs * dataPairs.[pairId][0].size() / batchSize;

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

        mpGraph->forward();
        std::shared_ptr<Tensor<double>> loss = mpGraph->getOutput(mLossIndex);
        
        std::cout << "Batch: " << iteration << " Training-error: " << loss->at(0);

        std::vector<std::shared_ptr<Tensor<double>>> gradientTable = mpGraph->backprop(Module::getLearnableParameters()); // backpropagation
        
        std::visit([gradientTable, batchSize](auto&& arg) {
            arg.update(gradientTable, batchSize);}, optimizer);

        if ( dataPairs[pairId][2].size() )
        {
            mDataPairs[pairId].first->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationData));
            mDataPairs[pairId].second->getData() = std::make_shared<Tensor<double>>(Matrix<double>(validationLabel));

            mpGraph->forward();
            std::shared_ptr<Tensor<double>> validationLoss = mpGraph->getOutput(mLossIndex);
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


void Model::train(Vector2D const trainData, Vector2D const trainLabel, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer)
{
    std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> dataPairs;
    if (mDataPairs.find(0) == mDataPairs.end())
    { // check if graph has an input pair for id 0
        throw std::runtime_error("no access point for data/label pair with id 0 found");
    }
    dataPairs[0] = std::make_pair(trainData, trainLabel);
    train(dataPairs, epochs, batchSize, optimizer, false);
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

        mpGraph->forward();
        std::shared_ptr<Tensor<double>> loss = mpGraph->getOutput(mLossIndex);
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