#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "model.hpp"

/**
 * @brief the sequential model is intended for creating a sequential neural network. It's a container for the modules.
*/
class SequentialModel : public Model
{
    std::vector<std::shared_ptr<Module>> mModules; // storing the modules in the order they were added
    std::shared_ptr<Variable> mpDataInputVariable; // storing the input variable
    std::shared_ptr<Variable> mpLabelInputVariable; // storing the label input variable
    std::shared_ptr<Variable> mpOutputVariable; // storing the output variable


protected:

    SequentialModel(std::vector<ModuleVariant> layers, std::uint32_t id);

};

SequentialModel::SequentialModel(std::vector<ModuleVariant> layers, std::uint32_t id)
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

    mLossIndex = GRAPH->addOutput(modules.back()->output());
    if (mDataPairs.find(id) != mDataPairs.end())
    {
        throw std::runtime_error("id already exists");
    }
    // add error checks in the future
    std::shared_ptr<Input> input = std::dynamic_pointer_cast<Input>(modules.front());
    std::shared_ptr<Cost> output = std::dynamic_pointer_cast<Cost>(modules.back());
    mDataPairs[id] = std::make_pair(input->data(), output->target());
}

#endif // SEQUENTIAL_HPP