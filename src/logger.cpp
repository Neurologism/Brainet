//
// Created by servant-of-scietia on 21.09.24.
//

#include "logger.hpp"

// std::cout << " \t \"validation_loss\": " << validationLoss->at(0) << ",\n";
// std::cout << " \t \"validation_surrogate_loss\": " << validationSurrogateLoss->at(0) << "\n";
// std::cout << "}"<< std::endl;


bool Logger::msJsonFormat = false;

double Logger::mMinimumLoss = std::numeric_limits<double>::max();
double Logger::mMinimumSurrogateLoss = std::numeric_limits<double>::max();
std::uint32_t Logger::mIteration = 0;

double Logger::mMinimumValidationLoss = std::numeric_limits<double>::max();
double Logger::mMinimumValidationSurrogateLoss = std::numeric_limits<double>::max();
std::uint32_t Logger::mEpoch = 1;

void Logger::logIteration(const double &loss, const double &surrogateLoss)
{
    std::cout << std::setprecision(5) << std::fixed;

    mIteration++;
    if (loss < mMinimumLoss)
    {
        mMinimumLoss = loss;
    }
    if (surrogateLoss < mMinimumSurrogateLoss)
    {
        mMinimumSurrogateLoss = surrogateLoss;
    }

    if (msJsonFormat)
    {
        std::ios_base::sync_with_stdio(false);
        std::cout << "{\n";
        std::cout << " \t \"epoch\": " << mEpoch << ",\n";
        std::cout << " \t \"iteration\": " << mIteration << ",\n";
        std::cout << " \t \"loss\": " << loss << ",\n";
        std::cout << " \t \"surrogate_loss\": " << surrogateLoss << ",\n";
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "Epoch: " << mEpoch << " Iteration: " << mIteration << " Loss: " << loss << " Surrogate Loss: " << surrogateLoss << std::string(10, ' ') << std::flush;
        std::cout << "\r";
    }
}

void Logger::logEpoch(const double &validationLoss, const double &validationSurrogateLoss)
{
    std::cout << std::setprecision(5) << std::fixed;


    if (validationLoss < mMinimumValidationLoss)
    {
        mMinimumValidationLoss = validationLoss;
    }
    if (validationSurrogateLoss < mMinimumValidationSurrogateLoss)
    {
        mMinimumValidationSurrogateLoss = validationSurrogateLoss;
    }

    if (msJsonFormat)
    {
        std::ios_base::sync_with_stdio(false);
        std::cout << "{\n";
        std::cout << " \t \"epoch\": " << mEpoch << ",\n";
        std::cout << " \t \"validation_loss\": " << validationLoss << ",\n";
        std::cout << " \t \"validation_surrogate_loss\": " << validationSurrogateLoss << ",\n";
        std::cout << " \t \"minimum_loss\": " << mMinimumLoss << ",\n";
        std::cout << " \t \"minimum_surrogate_loss\": " << mMinimumSurrogateLoss << ",\n";
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "\r" << std::string(100, ' ') << "\r"; // clear the line
        std::cout << "Epoch: " << mEpoch << "\n";
        std::cout << "Validation Set: Loss: " << validationLoss << " Surrogate Loss: " << validationSurrogateLoss << "\n";
        std::cout << "Training Set: lowest Loss: " << mMinimumLoss << " lowest Surrogate Loss: " << mMinimumSurrogateLoss << "\n";
        std::cout << "-------------------------------------------------------------------------------------" << std::endl;
    }

    mEpoch++;
    mIteration = 0;
    mMinimumLoss = std::numeric_limits<double>::max();
    mMinimumSurrogateLoss = std::numeric_limits<double>::max();
}
