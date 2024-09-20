#ifndef IMAGETRANSFORMATION_HPP
#define IMAGETRANSFORMATION_HPP

#include "dependencies.hpp"
////////////////////////
// under maintenance
////////////////////////
/**
 * @brief Used to apply various transformations to an image.
 */
class ImageTransformation
{
    typedef std::vector<std::vector<double>> data_type;

    static data_type to_2D(const std::vector<double> & input, std::uint32_t width, std::uint32_t height);

public:
    /**
     * @brief apply a random sequence of transformations to the input image
     * @param input the input image
     * @return the transformed image
     */
    static data_type randomTransform(const data_type & input);

    /**
     * @brief apply a random rotation to the input image
     * @param input the input image
     * @return the rotated image
     */
    static data_type randomRotation(const data_type & input);

    /**
     * @brief apply a random translation to the input image
     * @param input the input image
     * @param translation_range the range of the translation
     * @return the translated image
     */
    static data_type randomTranslation(const data_type & input, std::uint32_t translation_range)

    /**
     * @brief apply a random scaling to the input image
     * @param input the input image
     * @param scaling_range scales the image by a random factor in the range [1/scaling_range, scaling_range]
     * @return the scaled image
     */
    static data_type randomScaling(const data_type & input, const std::uint32_t & scaling_range)

    /**
     * @brief apply a random shear to the input image
     * @param input the input image
     * @param shear_angle angle telling the extent of the shear
     * @return the sheared image
     */
    static data_type randomShear(const data_type & input, const double & shear_angle);

    /**
     * @brief flips the input image horizontally or vertically with a 50% chance
     * @param input the input image
     * @return the flipped image
     */
    static data_type randomFlip(const data_type & input);

    /**
     * @brief apply a random contrast to the input image
     * @param input the input image
     * @return the contrasted image
     */
    static data_type randomContrast(const data_type & input, const double & contrast_range);
};





#endif // IMAGETRANSFORMATION_HPP