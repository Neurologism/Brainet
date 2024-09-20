//
// Created by servant-of-scietia on 20.09.24.
//
#include "operation/weight_initialization/weight_initializer.hpp"

std::vector<double> WeightInitializer::createRandomVector()
{
    std::vector<double> output(mInputUnits * mOutputUnits);

    for (std::uint32_t i = 0; i < mInputUnits * mOutputUnits; i++)
    {
        output[i] = generate();
    }

    return output;
}