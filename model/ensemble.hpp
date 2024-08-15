#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP

#include "model.hpp"
#include "../module/ensemble_module.hpp"

/**
 * @brief the ensemble module is intended for creating an ensemble of models in the graph. This is useful for techniques like bagging or boosting.
 */
class Ensemble : public Model
{
    std::vector<std::shared_ptr<Model>> mModels;        // storing the models in the ensemble
    std::shared_ptr<EnsembleModule> mEnsembleModule;    // the module that averages the output of the models

public:
    Ensemble(std::vector<ModelVariant> models, CostVariant costFunction);

    ~Ensemble() = default;

    void addModel(ModelVariant model);

    // should be supported in the future, need a rewrite of the train function
    void train(std::vector<Vector2D> const & inputs, std::vector<Vector2D> const & labels, std::uint32_t const epochs, std::uint32_t const batchSize, OptimizerVariant optimizer, std::uint32_t const earlyStoppingIteration = 20, double split = 0.8){};

    void test(Vector2D const & input, Vector2D const & label);
};

Ensemble::Ensemble(std::vector<ModelVariant> models, CostVariant costFunction)
{
    std::vector<std::shared_ptr<Variable>> ensembleInputs;
    for ( std::uint32_t i = 0; i < models.size(); i++)
    {
        std::shared_ptr<Model> modelPtr = std::visit([](auto&& arg) {
            return std::shared_ptr<Model>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ModelVariant{models[i]});

        mModels.push_back(modelPtr);

        
        mInputVariables.insert(mInputVariables.end(), modelPtr->mInputVariables.begin(), modelPtr->mInputVariables.end());
        mLearnableVariables.insert(mLearnableVariables.end(), modelPtr->mLearnableVariables.begin(), modelPtr->mLearnableVariables.end());
        ensembleInputs.insert(ensembleInputs.end(), modelPtr->mOutputVariables.begin(), modelPtr->mOutputVariables.end());
        // mBackpropVariables and mLearnableVariables to support backpropagation
    }

    mEnsembleModule = std::make_shared<EnsembleModule>(ensembleInputs, costFunction);

    mTargetVariables.push_back(mEnsembleModule->getVariable(2));
    mOutputVariables.push_back(mEnsembleModule->getVariable(0));
    mLossVariables.push_back(mEnsembleModule->getVariable(1));
}

void Ensemble::addModel(ModelVariant model)
{
    std::shared_ptr<Model> modelPtr = std::visit([](auto&& arg) {
        return std::shared_ptr<Model>(std::make_shared<std::decay_t<decltype(arg)>>(arg));}, ModelVariant{model});

    mModels.push_back(modelPtr);

    throw std::runtime_error("Ensemble::addModel: not implemented yet");
}

void Ensemble::test(Vector2D const & input, Vector2D const & label)
{
    std::vector<Vector2D> inputs;
    for ( std::uint32_t i = 0; i < mModels.size(); i++)
    {
        inputs.push_back(input);
    }

    Model::test(inputs, {label});
}


#endif // ENSEMBLE_HPP