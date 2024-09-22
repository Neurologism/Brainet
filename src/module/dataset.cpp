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

bool Dataset::goodBatch(const std::uint32_t &batchSize) const
{
    return mIndex + batchSize < mTrainingData.size();
}

bool Dataset::hasValidationSet() const
{
    return !mValidationData.empty();
}

void Dataset::shuffleTrainingSet()
{
    std::vector<std::uint32_t> indices(mTrainingData.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::ranges::shuffle(indices, std::mt19937{std::random_device{}()});

    dataType shuffledTrainingData(mTrainingData.size());
    dataType shuffledTrainingLabels(mTrainingLabels.size());

    for (std::uint32_t i = 0; i < mTrainingData.size(); i++)
    {
        shuffledTrainingData[i] = mTrainingData[indices[i]];
        shuffledTrainingLabels[i] = mTrainingLabels[indices[i]];
    }

    mTrainingData = shuffledTrainingData;
    mTrainingLabels = shuffledTrainingLabels;
}



void Dataset::loadTrainingBatch(const std::uint32_t &batchSize)
{
    if (mIndex + batchSize > mTrainingData.size())
    {
        throw std::invalid_argument("The batch size is larger than the remaining size of the training set.");
    }
    dataType dataBatch;
    dataType labelBatch;

    while (dataBatch.size() < batchSize)
    {
        dataBatch.push_back(mTrainingData[mIndex]);
        labelBatch.push_back(mTrainingLabels[mIndex]);
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