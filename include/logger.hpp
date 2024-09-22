//
// Created by servant-of-scietia on 21.09.24.
//

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "dependencies.hpp"

class Logger
{
    static double mMinimumLoss;
    static double mMinimumSurrogateLoss;
    static std::uint32_t mIteration;

    static double mMinimumValidationLoss;
    static double mMinimumValidationSurrogateLoss;
    static std::uint32_t mEpoch;

public:
    static bool msJsonFormat;
    static void logIteration(const double &loss, const double &surrogateLoss);
    static void logEpoch(const double &validationLoss, const double &validationSurrogateLoss);
};

#endif //LOGGER_HPP
