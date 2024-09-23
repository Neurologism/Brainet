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

Dataset::Dataset(const dataType &trainingData, const dataType &trainingLabels, const dataType &testData, const dataType &testLabels, const std::string &name) : Module(name)
{
    if (trainingData.size() != trainingLabels.size()) throw std::runtime_error("data and labels have different sizes");
    if (testData.size() != testLabels.size()) throw std::runtime_error("data and labels have different sizes");

    mTrainingData = trainingData;
    mTrainingLabels = trainingLabels;
    mTestData = testData;
    mTestLabels = testLabels;

    mDataVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mLabelVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
}

bool Dataset::goodTrainingBatch(const std::uint32_t &batchSize) const
{
    return mIndex + batchSize < mTrainingIndices.size();
}

bool Dataset::hasValidationSet() const
{
    return !mValidationData.empty();
}

void Dataset::shuffleTrainingSet(const bool completeTrainingSet)
{
    mTrainingIndices.resize(mTrainingData.size() + (completeTrainingSet ? mValidationData.size() : 0));
    std::iota(mTrainingIndices.begin(), mTrainingIndices.end(), 0);
    std::ranges::shuffle(mTrainingIndices, std::mt19937(std::random_device()()));

    mIndex = 0;
}



void Dataset::loadTrainingBatch(const std::uint32_t &batchSize)
{
    if (mIndex + batchSize > mTrainingIndices.size())
    {
        throw std::invalid_argument("The batch size is larger than the remaining size of the training set.");
    }
    dataType dataBatch;
    dataType labelBatch;

    while (dataBatch.size() < batchSize)
    {
        if (mTrainingIndices[mIndex] < mTrainingData.size())
        {
            dataBatch.push_back(mTrainingData[mTrainingIndices[mIndex]]);
            labelBatch.push_back(mTrainingLabels[mTrainingIndices[mIndex]]);
        }
        else
        {
            dataBatch.push_back(mValidationData[mTrainingIndices[mIndex] - mTrainingData.size()]);
            labelBatch.push_back(mValidationLabels[mTrainingIndices[mIndex] - mTrainingData.size()]);
        }
        mIndex++;
    }

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