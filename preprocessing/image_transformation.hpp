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
     * @brief Convert a 1D vector to a 2D vector.
     * @param input The input 1D vector.
     * @return The output 2D vector.
     */
    data_type to_2D(const std::vector<double> & input, std::uint32_t width, std::uint32_t height)
    {
        data_type output(width, std::vector<double>(height, 0.0));

        for (std::uint32_t i = 0; i < width; i++)
        {
            for (std::uint32_t j = 0; j < height; j++)
            {
                output[i][j] = input[i * height + j];
            }
        }

        return output;
    }

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
        std::uniform_real_distribution<double> dis(0, 1);

        double angle = dis(gen) * 3.14159265358979323846;

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
     * @param translation_range the range of the translation
     * @return the translated image
     */
    data_type randomTranslation(const data_type & input, std::uint32_t translation_range)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::uint32_t> dis(-translation_range, translation_range);

        std::uint32_t x = dis(gen);
        std::uint32_t y = dis(gen);

        data_type translatedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                if (i + x >= 0 && i + x < input.size() && j + y >= 0 && j + y < input[i].size())
                {
                    translatedImage[i][j] = input[i + x][j + y];
                }
            }
        }

        return translatedImage;
    }

    /**
     * @brief apply a random scaling to the input image
     * @param input the input image
     * @param scaling_range scales the image by a random factor in the range [1/scaling_range, scaling_range]
     * @return the scaled image
     */
    data_type randomScaling(const data_type & input, const std::uint32_t & scaling_range)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(1 / scaling_range, scaling_range);

        double scalingFactor = dis(gen);

        data_type scaledImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        double centerX = input.size() / 2;
        double centerY = input[0].size() / 2;

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                double x = i - centerX;
                double y = j - centerY;

                double newX = x * scalingFactor + centerX;
                double newY = y * scalingFactor + centerY;

                if (newX >= 0 && newX < input.size() && newY >= 0 && newY < input[i].size())
                {
                    scaledImage[i][j] = input[static_cast<std::uint32_t>(newX)][static_cast<std::uint32_t>(newY)];
                }
            }
        }

        return scaledImage;
    }

    /**
     * @brief apply a random shear to the input image
     * @param input the input image
     * @param shear_angle angle telling the extent of the shear
     * @return the sheared image
     */
    data_type randomShear(const data_type & input, const double & shear_angle)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-shear_angle, shear_angle);

        double shearAngle = dis(gen);

        data_type shearedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        double centerX = input.size() / 2;
        double centerY = input[0].size() / 2;

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                double x = i - centerX;
                double y = j - centerY;

                double newX = x + shearAngle * y + centerX;
                double newY = y + shearAngle * x + centerY;

                if (newX >= 0 && newX < input.size() && newY >= 0 && newY < input[i].size())
                {
                    shearedImage[i][j] = input[static_cast<std::uint32_t>(newX)][static_cast<std::uint32_t>(newY)];
                }
            }
        }

        return shearedImage;
    }

    /**
     * @brief flips the input image horizontally or vertically with a 50% chance
     * @param input the input image
     * @return the flipped image
     */
    data_type randomFlip(const data_type & input)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::uint32_t> dis(0, 1);

        std::uint32_t vertical = dis(gen);

        data_type verticalFlippedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                if (vertical)
                {
                    verticalFlippedImage[i][j] = input[i][input[i].size() - j - 1];
                }
                else
                {
                    verticalFlippedImage[i][j] = input[i][j];
                }
            }
        }

        std::uint32_t horizontal = dis(gen);

        data_type flippedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                if (horizontal)
                {
                    flippedImage[i][j] = verticalFlippedImage[input.size() - i - 1][j];
                }
                else
                {
                    flippedImage[i][j] = verticalFlippedImage[i][j];
                }
            }
        }

        return flippedImage;
    }

    /**
     * @brief apply a random contrast to the input image
     * @param input the input image
     * @return the contrasted image
     */
    data_type randomContrast(const data_type & input, const double & contrast_range)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(1 / contrast_range, contrast_range);

        double contrastFactor = dis(gen);

        data_type contrastedImage(input.size(), std::vector<double>(input[0].size(), 0.0));

        for (std::uint32_t i = 0; i < input.size(); i++)
        {
            for (std::uint32_t j = 0; j < input[i].size(); j++)
            {
                contrastedImage[i][j] = input[i][j] * contrastFactor;
            }
        }

        return contrastedImage;
    }
};





#endif // IMAGETRANSFORMATION_HPP