#ifndef IMAGETRANSFORMATION_HPP
#define IMAGETRANSFORMATION_HPP

#include "../dependencies.hpp"

/**
 * @brief Used to apply various transformations to an image.
 */
namespace ImageTransformation
{
    typedef std::vector<std::vector<double>> data_type;

    /**
     * @brief apply a random sequence of transformations to the input image
     * @param input the input image
     * @return the transformed image
     */
    data_type randomTransform(const data_type & input);

    /**
     * @brief apply a random rotation to the input image
     * @param input the input image
     * @return the rotated image
     */
    data_type randomRotation(const data_type & input)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0, 360);

        double angle = dis(gen);

        data_type rotatedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        double centerX = input.size() / 2;
        double centerY = input[0].size() / 2;

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                double x = i - centerX;
                double y = j - centerY;

                double newX = x * std::cos(angle) - y * std::sin(angle) + centerX;
                double newY = x * std::sin(angle) + y * std::cos(angle) + centerY;

                if (newX >= 0 && newX < input.size() && newY >= 0 && newY < input[i].size())
                {
                    rotatedImage[i][j] = input[static_cast<std::uint32_t>(newX)][static_cast<std::uint32_t>(newY)];
                }
            }
        }

        return rotatedImage;
    }

    /**
     * @brief apply a random translation to the input image
     * @param input the input image
     * @return the translated image
     */
    data_type randomTranslation(const data_type & input);

    /**
     * @brief apply a random scaling to the input image
     * @param input the input image
     * @return the scaled image
     */
    data_type randomScaling(const data_type & input);

    /**
     * @brief apply a random shear to the input image
     * @param input the input image
     * @return the sheared image
     */
    data_type randomShear(const data_type & input);

    /**
     * @brief apply a random flip to the input image
     * @param input the input image
     * @return the flipped image
     */
    data_type randomFlip(const data_type & input);

    /**
     * @brief apply a random contrast to the input image
     * @param input the input image
     * @return the contrasted image
     */
    data_type randomContrast(const data_type & input);
};





#endif // IMAGETRANSFORMATION_HPP