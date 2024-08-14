#ifndef SPLIT_HPP
#define SPLIT_HPP

#include "../dependencies.hpp"

namespace preprocessing
{
    typedef std::vector<std::vector<double>> dataType;


    /**
     * @brief This function splits the data into a training and a validation set.
     * @param data The data to split.
     * @param ratio The ratio of the training set.
     */
    void splitData(dataType const & input, dataType const & target, double const & ratio, dataType & trainInput, dataType & validationInput, dataType & trainTarget, dataType & validationTarget)
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

        std::cout << "Data split into " << trainInput.size() << " training samples and " << validationInput.size() << " validation samples." << std::endl;
    }
} // namespace preprocessing
#endif // SPLIT_HPP