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

    SequentialModel(Input input_layer, std::vector<ModuleVariant> hidden_layers, Output output_layer, Cost cost_function);

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
    inputLayer->addOutput(hiddenModules.front()->input());
    hiddenModules.front()->addInput(inputLayer->output(), inputLayer->getUnits());
    
    for(std::uint32_t i = 0; i < hiddenModules.size() - 1; i++)
    {
        hiddenModules[i]->addOutput(hiddenModules[i+1]->input());
        hiddenModules[i+1]->addInput(hiddenModules[i]->output(),hiddenModules[i]->getUnits());
    }

    hiddenModules.back()->addOutput(outputLayer->input());
    outputLayer->addInput(hiddenModules.back()->output(), hiddenModules.back()->getUnits());

    outputLayer->addOutput(costFunction->input());
    costFunction->addInput(outputLayer->output(), outputLayer->getUnits());


    // store modules
    mModules.push_back(inputLayer);
    mModules.insert(mModules.end(), hiddenModules.begin(), hiddenModules.end());
    mModules.push_back(outputLayer);
    mModules.push_back(costFunction);

    // store special variables

    for
    mInputVariables.push_back(inputLayer->input());
    mOutputVariables.push_back(outputLayer->output(0));
    mLossVariables.push_back(costFunction->output(0));
    mTargetVariables.push_back(costFunction->input(1));
    

}

#endif // SEQUENTIAL_HPP