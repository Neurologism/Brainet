#ifndef MODEL_HPP
#define MODEL_HPP

#include "graph.hpp"
#include "module/module_variant.hpp"
#include "optimizer/optimizer_variant.hpp"
#include "module/dataset.hpp"

/**
 * @brief The Model class is intended to be used as a base class for all models.
 * @details The Model class provides the basic functionality to train and test a model.
 * It is intended to be extended by different model classes like SequentialModel or Ensemble that 
 * provide simple interfaces to create and train models and offer a constructor to create the model.
 * @note The Model class is abstract and cannot be instantiated.
 */
class Model
{
    // everything needed for the graph
    std::vector<std::shared_ptr<Variable>> mLearnableVariables; // all variables that can be learned by the learning algorithm
    std::vector<std::shared_ptr<Variable>> mGradientVariables;  // all variables that are used as starting point for the backpropagation and are leafs of the model subgraph
    std::vector<std::shared_ptr<Variable>> mLossVariables;      // all variables that are used as output for the loss

    std::vector<std::shared_ptr<Module>> mModules; // all modules of the model
    std::map<std::string, std::shared_ptr<Module>> mModuleMap; // map to access modules by name

    bool earlyStopping(const std::uint32_t &epoch, std::uint32_t &bestEpoch, const std::uint32_t &earlyStoppingPatience, const double &error, double &bestError, std::vector<std::shared_ptr<Tensor<double>>> &bestParameters);

public:

    std::shared_ptr<Module> addModule(const ModuleVariant &module);

    void addSequential(const std::vector<ModuleVariant> &modules);

    static void connectModules(const std::shared_ptr<Module> &startModule, const std::shared_ptr<Module> &endModule);

    void connectModules(const std::string &startModule, const std::string &endModule);

    /**
     * @brief function to train the model
     * @param dataset the dataset to train the model
     * @param inputModule the name of the input module
     * @param lossModule the name of the loss module
     * @param epochs the number of epochs
     * @param batchSize the size of the batch
     * @param optimizer the optimizer to use
     * @param earlyStoppingPatience the number of epochs to wait before stopping the training
     */
    void train(Dataset &dataset, const std::string& inputModule, const std::string& lossModule, const std::uint32_t &epochs, const std::uint32_t &batchSize, OptimizerVariant optimizer, const std::uint32_t &earlyStoppingPatience);

    /**
     * @brief function to test the model
     * @note the function will print the error of the model
     */
    void test(Dataset &dataset, const std::string& inputModule, const std::string& lossModule);

    friend class Ensemble;
};

#endif // MODEL_HPP