//
// Created by servant-of-scietia on 20.09.24.
//

#include "preprocessing/preprocessing.hpp"

void Preprocessing::createBatch(const dataType &data, const dataType &labels, const std::uint32_t &batchSize, dataType &dataBatch, dataType &labelBatch)
{
    // generate random batch
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<std::int32_t>(data.size()) - 1);

    for (std::uint32_t i = 0; i < batchSize; i++)
    {
        const std::uint32_t randomIndex = dis(gen);
        dataBatch.push_back(data[randomIndex]);
        labelBatch.push_back(labels[randomIndex]);
    }
}

void Preprocessing::createBatch(const dataType &data, const dataType &labels, const std::uint32_t &batchSize, dataType &dataBatch, dataType &labelBatch, const double &mean, const double &stddev)
{
    createBatch(data, labels, batchSize, dataBatch, labelBatch);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);

    for (auto & i : dataBatch)
    {
        for (double & j : i)
        {
            j += dist(gen);
        }
    }
}

Preprocessing::dataType Preprocessing::normalize(dataType const & input)
{
    dataType normalizedData(input.size(), std::vector<double>(input[0].size(), 0.0));

    double max = std::numeric_limits<double>::min();

    for (std::uint32_t i = 0; i < input.size(); i++)
    {
        for (std::uint32_t j = 0; j < input[i].size(); j++)
        {
            if (input[i][j] > max)
            {
                max = input[i][j];
            }
        }
    }

    for (std::uint32_t i = 0; i < input.size(); i++)
    {
        for (std::uint32_t j = 0; j < input[i].size(); j++)
        {
            normalizedData[i][j] = input[i][j] / max;
        }
    }

    return normalizedData;
}

void Preprocessing::splitData(dataType const & input, dataType const & target, double const & ratio, dataType & trainInput, dataType & validationInput, dataType & trainTarget, dataType & validationTarget)
{
    if (ratio < 0.0 || ratio > 1.0)
    {
        throw std::invalid_argument("The ratio must be between 0 and 1.");
    }
    if (input.size() != target.size())
    {
        throw std::invalid_argument("The input and target sizes must be the same.");
    }

    trainInput = {};
    validationInput = {};
    trainTarget = {};
    validationTarget = {};

    std::uint32_t split_index = std::round(input.size() * ratio);

    std::vector<std::uint32_t> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    for (std::uint32_t i = 0; i < split_index; i++)
    {
        trainInput.push_back(input[indices[i]]);
        trainTarget.push_back(target[indices[i]]);
    }

    for (std::uint32_t i = split_index; i < input.size(); i++)
    {
        validationInput.push_back(input[indices[i]]);
        validationTarget.push_back(target[indices[i]]);
    }
}