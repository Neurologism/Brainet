//
// Created by servant-of-scietia on 13.09.24.
//

#ifndef DATASET_HPP
#define DATASET_HPP

#include "module.hpp"
#include "preprocessing/preprocessing.hpp"

/**
 * @brief The Dataset class is used to store the data for storing the training and test data of datasets.
 */
class Dataset final : public Module
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

    std::uint32_t mIndex = 0;

public:
    Dataset(const dataType &trainingData, const dataType &trainingLabels, const double &validationSplit, const dataType &testData, const dataType &testLabels, const std::string &name = "");
    Dataset(const dataType &trainingData, const dataType &trainingLabels, const dataType &testData, const dataType &testLabels, const std::string &name = "");


    [[nodiscard]] bool goodBatch(const std::uint32_t &batchSize) const;
    [[nodiscard]] bool hasValidationSet() const;

    void shuffleTrainingSet();
    void loadTrainingBatch(const std::uint32_t &batchSize);
    void loadValidationSet() const;
    void loadTestSet() const;

    std::vector<std::shared_ptr<Variable>> getInputs() override;
    std::vector<std::shared_ptr<Variable>> getOutputs() override;
    std::vector<std::shared_ptr<Variable>> getLearnableVariables() override;
    std::vector<std::shared_ptr<Variable>> getGradientVariables() override;
};

#endif //DATASET_HPP
