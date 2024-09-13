//
// Created by servant-of-scietia on 13.09.24.
//

#ifndef DATASET_HPP
#define DATASET_HPP

#include "module.hpp"
#include "../preprocessing/split.hpp"

/**
 * @brief The Dataset class is used to store the data for storing the training and test data of datasets.
 */
class Dataset final : Module
{
    std::shared_ptr<Variable> mDataVariable; // storing the data
    std::shared_ptr<Variable> mLabelVariable; // storing the labels

    typedef std::vector<std::vector<double>> dataType;
    dataType mTrainingData;
    dataType mTrainingLabels;
    dataType mValidationData;
    dataType mValidationLabels;
    dataType mTestData;
    dataType mTestLabels;

    // not implemented
    void addInput(const std::shared_ptr<Variable> &input, const std::uint32_t &inputSize) override {}
    void addOutput(const std::shared_ptr<Variable> &output) override {}
public:
    Dataset(const dataType &trainingData, const dataType &trainingLabels, const std::uint32_t &validationSplit, const dataType &testData, const dataType &testLabels, const std::string &name);

    std::shared_ptr<Variable> getVariable(std::uint32_t index) override;

    [[nodiscard]] std::uint32_t getTrainingSize() const { return mTrainingData.size(); }
    [[nodiscard]] std::uint32_t getValidationSize() const { return mValidationData.size(); }

    void loadTrainingBatch(std::uint32_t batchSize);
    void loadValidationBatch(std::uint32_t batchSize);
    void loadTestSet();
};


inline Dataset::Dataset(const dataType &trainingData, const dataType &trainingLabels, const std::uint32_t &validationSplit, const Dataset::dataType &testData, const Dataset::dataType &testLabels, const std::string &name) : Module(name)
{
    if (trainingData.size() != trainingLabels.size()) throw std::runtime_error("data and labels have different sizes");
    if (testData.size() != testLabels.size()) throw std::runtime_error("data and labels have different sizes");

    preprocessing::splitData(trainingData, trainingLabels, validationSplit, mTrainingData, mTrainingLabels, mValidationData, mValidationLabels);
    mTestData = testData;
    mTestLabels = testLabels;

    mDataVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
    mLabelVariable = GRAPH->addVariable(std::make_shared<Variable>(Variable(nullptr, {}, {})));
}

void Dataset::loadTrainingBatch(std::uint32_t batchSize)
{
    dataType dataBatch;
    dataType labelBatch;
    preprocessing::createBatch(mTrainingData, mTrainingLabels, batchSize, dataBatch, labelBatch);
    mDataVariable->setData(dataBatch);
    mLabelVariable->setData(labelBatch);
}

#endif //DATASET_HPP
