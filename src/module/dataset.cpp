//
// Created by servant-of-scietia on 20.09.24.
//
#include "module/dataset.hpp"

Dataset::Dataset(const dataType &trainingData, const dataType &trainingLabels, const double &validationSplit, const Dataset::dataType &testData, const Dataset::dataType &testLabels, const std::string &name) : Module(name)
{
    if (trainingData.size() != trainingLabels.size()) throw std::runtime_error("data and labels have different sizes");
    if (testData.size() != testLabels.size()) throw std::runtime_error("data and labels have different sizes");

    Preprocessing::splitData(trainingData, trainingLabels, validationSplit, mTrainingData, mValidationData, mTrainingLabels, mValidationLabels);
    mTestData = testData;
    mTestLabels = testLabels;

    mDataVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mLabelVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
}

void Dataset::loadTrainingBatch(const std::uint32_t batchSize) const
{
    dataType dataBatch;
    dataType labelBatch;
    Preprocessing::createBatch(mTrainingData, mTrainingLabels, batchSize, dataBatch, labelBatch);
    mDataVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(dataBatch)));
    mLabelVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(labelBatch)));
}

void Dataset::loadValidationSet() const
{
    mDataVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(mValidationData)));
    mLabelVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(mValidationLabels)));
}

void Dataset::loadTestSet() const
{
    mDataVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(mTestData)));
    mLabelVariable->setData(std::make_shared<Tensor<double>>(Matrix<double>(mTestLabels)));
}

std::vector<std::shared_ptr<Variable>> Dataset::getInputs()
{
    return {};
}

std::vector<std::shared_ptr<Variable>> Dataset::getOutputs()
{
    return {mDataVariable, mLabelVariable};
}

std::vector<std::shared_ptr<Variable>> Dataset::getLearnableVariables()
{
    return {};
}

std::vector<std::shared_ptr<Variable>> Dataset::getGradientVariables()
{
    return {};
}