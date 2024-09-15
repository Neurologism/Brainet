#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "model.hpp"

/**
 * @brief the sequential model is intended for creating a sequential neural network. It is a subclass of the Model class.
*/
class SequentialModel : public Model
{
    typedef std::vector<std::vector<double>> Vector2D;
public:
    /**
     * @brief create a new sequential neural network model
     * @param input_layer the input layer of the model.
     * @param hidden_layers the hidden layers of the model.
     * @param output_layer the output layer of the model.
     * @param cost_function the cost function of the model.
     * @note calls the __init__ function of every module with __init__({previousModule->getVariable(1)}, {nextModule->getVariable(0)})
     */
    SequentialModel(Input input_layer, std::vector<ModuleVariant> hidden_layers, Output output_layer);

    ~SequentialModel() = default;

    void train(Vector2D const & input, Vector2D const & label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStopping = 20, double split = 0.8);


    void test(Vector2D const & data, Vector2D const & label);
};

SequentialModel::SequentialModel(Input input_layer, std::vector<ModuleVariant> hidden_layers, Output output_layer)
{
    // convert modules to shared pointers
    std::shared_ptr<Input> inputLayer = std::make_shared<Input>(input_layer);
    std::vector<std::shared_ptr<Module>> hiddenModules;
    std::shared_ptr<Output> outputLayer = std::make_shared<Output>(output_layer);

    for (ModuleVariant& layer : hidden_layers) {
        std::shared_ptr<Module> modulePtr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ModuleVariant{layer});
        hiddenModules.push_back(modulePtr);
    }
    
    // store modules
    mModules.push_back(inputLayer);
    mModules.insert(mModules.end(), hiddenModules.begin(), hiddenModules.end());
    mModules.push_back(outputLayer);

    // connect Layers in Graph
    // assumes for all modules that getVariable(0) is the input variable and getVariable(1) is the output variable

    connectModules(inputLayer, hiddenModules[0]);
    for (std::uint32_t i = 0; i < hiddenModules.size() - 1; i++)
    {
        connectModules(hiddenModules[i], hiddenModules[i+1]);
    }
    connectModules(hiddenModules.back(), outputLayer);

    FullyConnected* output = dynamic_cast<FullyConnected*>(outputLayer.get());
    if (output != nullptr) 
    {
        output->createWeightMatrix(hiddenModules.back()->getUnits());
    }



}

void SequentialModel::train(Vector2D const & design_matrix, Vector2D const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStopping, double split)
{
    Model::train({design_matrix}, {labels}, epochs, batchSize, optimizer, earlyStopping, split);
}

void SequentialModel::test(Vector2D const & data, Vector2D const & label)
{
    Model::test({data}, {label});
}

#endif // SEQUENTIAL_HPP