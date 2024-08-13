#ifndef SPLIT_HPP
#define SPLIT_HPP

#include "../dependencies.hpp"

/**
 * @brief This function splits the data into a training and a validation set.
 * @param data The data to split.
 * @param ratio The ratio of the training set.
 * @return A pair of the training and validation set.
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> split(std::vector<std::vector<double>> const & data, double const ratio)
{
    if (ratio < 0.0 || ratio > 1.0)
    {
        throw std::invalid_argument("The ratio must be between 0 and 1.");
    }
    
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> result;
    std::uint32_t split_index = std::round(data.size() * ratio);

    std::vector<std::vector<double>> shuffled_data = data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);

    result.first = std::vector<std::vector<double>>(shuffled_data.begin(), shuffled_data.begin() + split_index);
    result.second = std::vector<std::vector<double>>(shuffled_data.begin() + split_index, shuffled_data.end());

    return result;
}

#endif // SPLIT_HPP