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
        Module::set_graph(mpGraph); // set the static graph pointer in the module class
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
    void sequential(std::vector<Module_VARIANT> layers, std::uint32_t id = 0);

    /**
     * @brief This function creates a sequential neural network.
     * @param layers The layers of the neural network.
     * @param norm The norm to use for regularization.
     * @param id The id of the data/label pair.
     */
    void sequential(std::vector<Module_VARIANT> layers, NormVariant norm, std::uint32_t id = 0);

    /**
     * @brief This function trains the model. It uses the backpropagation algorithm to update the learnable parameters. 
     * @param dataPairs Map distributing the data/label pairs to the input/output nodes according to their id. id : (data, label)
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param learning_rate The learning rate.
     */
    void train(std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> const & dataPairs, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer);
    /**
     * @brief Shortcut for training a model with only one data/label pair. Assumes the id is 0.
     * @param data The data.
     * @param label The label.
     * @param epochs The number of epochs.
     * @param batchSize The batch size.
     * @param learning_rate The learning rate.
     */
    void train(Vector2D const data, Vector2D const label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer);

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
    Module::set_graph(mpGraph);
}

void Model::sequential(std::vector<Module_VARIANT> layers, std::uint32_t id)
{
    std::vector<std::shared_ptr<Module>> modules;

    for (Module_VARIANT& layer : layers) {
        std::shared_ptr<Module> cluster_ptr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, Module_VARIANT{layer});
        modules.push_back(cluster_ptr);
    }
    
    for(std::uint32_t i = 0; i < layers.size() - 1; i++)
    {
        modules[i]->add_output(modules[i+1]->input());
        modules[i+1]->add_input(modules[i]->output(),modules[i]->getUnits());
    }

    mLossIndex = mpGraph->add_output(modules.back()->output());
    if (mDataPairs.find(id) != mDataPairs.end())
    {
        throw std::runtime_error("id already exists");
    }
    // add error checks in the future
    std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(modules.front());
    std::shared_ptr<Cost> output = std::dynamic_pointer_cast<Cost>(modules.back());
    mDataPairs[id] = std::make_pair(input->data(), output->target());
}

void Model::sequential(std::vector<Module_VARIANT> layers, NormVariant norm, std::uint32_t id)
{
    Dense::set_default_norm(norm);
    sequential(layers, id);
}


void Model::train(std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> const & dataPairs, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer)
{
    if ( dataPairs.size() != 1)
    {
        throw std::runtime_error("Only one data/label pair is currently supported");
    }

    const std::uint32_t trainingIterations = epochs; // adjust this value later to train for epochs

    // this should be replaced by a more sophisticated training algorithm
    for(std::uint32_t iteration = 0; iteration < trainingIterations; iteration++)
    {
        Vector2D batchData;
        Vector2D batchLabel;

        Vector2D data = dataPairs.at(0).first;
        Vector2D label = dataPairs.at(0).second;

        // generate random batch
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        for (std::uint32_t i = 0; i < batchSize; i++)
        {
            std::uint32_t randomIndex = dis(gen);
            batchData.push_back(data[randomIndex]);
            batchLabel.push_back(label[randomIndex]);
        }
        

        mDataPairs[0].first->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(batchData));
        mDataPairs[0].second->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(batchLabel));

        mpGraph->forward();
        std::shared_ptr<Tensor<double>> loss = mpGraph->get_output(mLossIndex);
        std::cout << "Batch: " << iteration << " Loss: " << loss->at(0) << std::endl; // print loss
        std::vector<std::shared_ptr<Tensor<double>>> gradientTable = mpGraph->backprop(Module::get_learnable_parameters()); // backpropagation
        
        std::visit([gradientTable, batchSize](auto&& arg) {
            arg.update(gradientTable, batchSize);}, optimizer);
    }
}


void Model::train(Vector2D const data, Vector2D const label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer)
{
    std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> dataPairs;
    if (mDataPairs.find(0) == mDataPairs.end())
    {
        throw std::runtime_error("Assumed id 0, but no data/label pair with id 0 found");
    }
    dataPairs[0] = std::make_pair(data, label);
    train(dataPairs, epochs, batchSize, optimizer);
}


void Model::test(std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> const & dataPairs)
{
    for (auto const & dataPair : dataPairs)
    {
        mDataPairs[dataPair.first].first->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(dataPair.second.first));
        mDataPairs[dataPair.first].second->get_data() = std::make_shared<Tensor<double>>(Matrix<double>(dataPair.second.second));

        mpGraph->forward();
        std::shared_ptr<Tensor<double>> loss = mpGraph->get_output(mLossIndex);
        std::cout << "Test Loss: " << loss->at(0) << std::endl; // print loss
    }
}


void Model::test(Vector2D const data, Vector2D const label)
{
    std::map<std::uint32_t, std::pair<Vector2D, Vector2D>> dataPairs;
    if (mDataPairs.find(0) == mDataPairs.end())
    {
        throw std::runtime_error("Assumed id 0, but no data/label pair with id 0 found");
    }
    dataPairs[0] = std::make_pair(data, label);
    test(dataPairs);
}

#endif // MODEL_HPP