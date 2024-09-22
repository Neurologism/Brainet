//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "dependencies.hpp"

class Preprocessing
{
    typedef std::vector<std::vector<double>> dataType;
public:
    static void createBatch(const dataType &data, const dataType &labels, const std::uint32_t &batchSize, dataType &dataBatch, dataType &labelBatch);
    static void addNoise(dataType &data, const double &mean, const double &stddev);
    static dataType normalize(dataType const & input);
    static void splitData(dataType const & input, dataType const & target, double const & ratio, dataType & trainInput, dataType & validationInput, dataType & trainTarget, dataType & validationTarget);
};

#endif //PREPROCESSING_HPP
