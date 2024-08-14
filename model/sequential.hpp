#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "model.hpp"

/**
 * @brief the sequential model is intended for creating a sequential neural network. It's a container for the hiddenModules.
*/
class SequentialModel : public Model
{
    typedef std::vector<std::vector<double>> Vector2D;

    std::vector<std::shared_ptr<Module>> mModules; // storing the hiddenModules in the order they were added



protected:


public:
    /**
     * @brief add a sequential model of a neural network to the graph.
     * @param input_layer the input layer of the model.
     * @param hidden_layers the hidden layers of the model.
     * @param output_layer the output layer of the model.
     * @param cost_function the cost function of the model.
     * @note calls the __init__ function of every module with __init__({previousModule->getVariable(1)}, {nextModule->getVariable(0)})
     */
    SequentialModel(Input input_layer, std::vector<ModuleVariant> hidden_layers, Output output_layer, Cost cost_function);

    ~SequentialModel() = default;

    void train(Vector2D const & input, Vector2D const & label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, double split, std::uint32_t const earlyStopping);


    void test(Vector2D const & data, Vector2D const & label);
};

SequentialModel::SequentialModel(Input input_layer, std::vector<ModuleVariant> hidden_layers, Output output_layer, Cost cost_function)
{
    // convert modules to shared pointers
    std::shared_ptr<Input> inputLayer = std::make_shared<Input>(input_layer);
    std::vector<std::shared_ptr<Module>> hiddenModules;
    std::shared_ptr<Output> outputLayer = std::make_shared<Output>(output_layer);
    std::shared_ptr<Cost> costFunction = std::make_shared<Cost>(cost_function);

    for (ModuleVariant& layer : hidden_layers) {
        std::shared_ptr<Module> modulePtr = std::visit([](auto&& arg) {
        // Assuming all types in the variant can be dynamically casted to OPERATION*
        return std::shared_ptr<Module>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ModuleVariant{layer});
        hiddenModules.push_back(modulePtr);
    }

    // connect Layers in Graph
    // assumes for all modules that getVariable(0) is the input variable and getVariable(1) is the output variable

    inputLayer->__init__({}, {hiddenModules.front()->getVariable(0)});
    
    for(std::uint32_t i = 0; i < hiddenModules.size() - 1; i++)
    {
        if ( i == 0)
        {
            hiddenModules[i]->__init__({inputLayer->getVariable(1)}, {hiddenModules[i+1]->getVariable(0)});
        }
        else if (i == hiddenModules.size() - 1)
        {
            hiddenModules[i]->__init__({hiddenModules[i-1]->getVariable(1)}, {outputLayer->getVariable(0)});
        }
        else
        {
            hiddenModules[i]->__init__({hiddenModules[i-1]->getVariable(1)}, {hiddenModules[i+1]->getVariable(0)});
        }
    }

    outputLayer->__init__({hiddenModules.back()->getVariable(1)}, {costFunction->getVariable(0)});

    costFunction->__init__({outputLayer->getVariable(1)}, {});


    // store modules
    mModules.push_back(inputLayer);
    mModules.insert(mModules.end(), hiddenModules.begin(), hiddenModules.end());
    mModules.push_back(outputLayer);
    mModules.push_back(costFunction);


    // store special variables
    for (auto & module : hiddenModules)
    {
        if (dynamic_cast<Dense*>(module.get()) != nullptr) {
            // module is an object of class Dense
            mLearnableVariables.push_back(module->getVariable(2));      // weight matrix
            mBackpropVariables.push_back(module->getVariable(3));       // norm
        }
    }
    mLearnableVariables.push_back(outputLayer->getVariable(2));         // weight matrix

    mInputVariables.push_back(inputLayer->getVariable(0));              // input

    mTargetVariables.push_back(costFunction->getVariable(2));           // target

    mOutputVariables.push_back(outputLayer->getVariable(1));            // model output

    mLossVariables.push_back(costFunction->getVariable(1));             // cost function

    mBackpropVariables.push_back(outputLayer->getVariable(3));          // norm
    mBackpropVariables.push_back(costFunction->getVariable(1));         // cost function

}

void SequentialModel::train(Vector2D const & input, Vector2D const & label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, double split, std::uint32_t const earlyStopping)
{
    train({input}, {label}, epochs, batchSize, optimizer, split, earlyStopping);
}

void SequentialModel::test(Vector2D const & data, Vector2D const & label)
{
    test({data}, {label});
}

#endif // SEQUENTIAL_HPP