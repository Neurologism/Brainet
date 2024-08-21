#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include "../dependencies.hpp"

/**
 * @brief Used to normalize the input data.
 */
namespace preprocessing
{
    typedef std::vector<std::vector<double>> dataType;

    /**
     * @brief Normalize the input data. Maps the data to the range [0, 1].
     * @param input The input data.
     * @return The normalized data.
     */
    dataType normalize(dataType const & input)
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

} // namespace preprocessing

#endif // NORMALIZE_HPP