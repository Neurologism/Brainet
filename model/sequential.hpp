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

    inputLayer->__init__({}, {hiddenModules.front()->getVariable(0)});
    
    for(std::uint32_t i = 1; i < mModules.size()-1; i++)
    {
        mModules[i]->__init__({mModules[i-1]->getVariable(1)}, {mModules[i+1]->getVariable(0)});
    }

    outputLayer->__init__({hiddenModules.back()->getVariable(1)}, {});


    // load weight matrices

    for (std::uint32_t i = 0; i < hiddenModules.size(); i++)  
    {
        FullyConnected* dense = dynamic_cast<FullyConnected*>(hiddenModules[i].get());
        if (dense != nullptr) 
        {
            if (i == 0)
            {
                dense->createWeightMatrix(inputLayer->getUnits());
            }
            else
            {
                dense->createWeightMatrix(hiddenModules[i-1]->getUnits());
            }
        }
    }

    FullyConnected* output = dynamic_cast<FullyConnected*>(outputLayer.get());
    if (output != nullptr) 
    {
        output->createWeightMatrix(hiddenModules.back()->getUnits());
    }


    // store special variables
    for (auto & module : hiddenModules)
    {
        if (dynamic_cast<Dense*>(module.get()) != nullptr) 
        {
            // module is an object of class Dense
            mLearnableVariables.push_back(module->getVariable(2));      // weight matrix
            try
            {
                mBackpropVariables.push_back(module->getVariable(3));   // norm
            }
            catch(const std::exception& e)
            {
                // no norm variable
            }
        }
    }
    mLearnableVariables.push_back(outputLayer->getVariable(2));         // weight matrix

    mInputVariables.push_back(inputLayer->getVariable(0));              // input

    mOutputVariables.push_back(outputLayer->getVariable(1));            // model output

    try
    {
        mTargetVariables.push_back(outputLayer->getVariable(6));        // target
        mLossVariables.push_back(outputLayer->getVariable(5));          // loss
        mLossVariables.push_back(outputLayer->getVariable(4));          // surrogate loss
        mBackpropVariables.push_back(outputLayer->getVariable(4));      // surrogate loss
    }
    catch(const std::exception& e)
    {
        // no cost module
    }
    try
    {
        mBackpropVariables.push_back(outputLayer->getVariable(3));      // norm
    }
    catch(const std::exception& e)
    {
        // no norm variable
    }
}

void SequentialModel::train(Vector2D const & input, Vector2D const & label, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStopping, double split)
{
    Model::train({input}, {label}, epochs, batchSize, optimizer, earlyStopping, split);
}

void SequentialModel::test(Vector2D const & data, Vector2D const & label)
{
    Model::test({data}, {label});
}

#endif // SEQUENTIAL_HPP