#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "modul.hpp"

/**
 * @brief the sequential model is intended for creating a sequential neural network. It's a container for the modules.
*/
class SequentialModel : public Model
{
    std::vector<std::shared_ptr<Module>> mModules; // storing the modules in the order they were added
    std::shared_ptr<Variable> mpDataInputVariable; // storing the input variable
    std::shared_ptr<Variable> mpLabelInputVariable; // storing the label input variable
    std::shared_ptr<Variable> mpOutputVariable; // storing the output variable

