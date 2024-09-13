//
// Created by servant-of-scietia on 13.09.24.
//

#ifndef BATCH_HPP
#define BATCH_HPP

#include "../dependencies.hpp"

namespace preprocessing
{
    typedef std::vector<std::vector<double>> dataType;

    /**
     * @brief This function creates a batch from the input and target data.
     * @param data The input data.
     * @param labels The target data.
     * @param batchSize The size of the batch.
     * @param dataBatch The 2D vector to store the input batch.
     * @param labelBatch The 2D vector to store the target batch.
     */
    inline void createBatch(const dataType &data, const dataType &labels, std::uint32_t batchSize, dataType &dataBatch, dataType &labelBatch)
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

    /**
     * @brief This function creates a batch from the input and target data and adds noise to the input data.
     * @param data The input data.
     * @param labels The target data.
     * @param batchSize The size of the batch.
     * @param dataBatch The 2D vector to store the input batch.
     * @param labelBatch The 2D vector to store the target batch.
     * @param mean The mean of the noise.
     * @param stddev The standard deviation of the noise.
     */
    inline void createBatch(const dataType &data, const dataType &labels, const std::uint32_t &batchSize, dataType &dataBatch, dataType &labelBatch, const double &mean, const double &stddev)
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

}


#endif //BATCH_HPP
