#ifndef IDXREADER_HPP
#define IDXREADER_HPP

#include "../dependencies.hpp"

/**
 * @brief This function reads an IDX file and returns the data as a 2D vector.
 * @param path The path to the IDX file.
 * @return The data as a 2D vector.
 */
std::vector<std::vector<double>> read_idx(const std::string path)
{
    typedef std::vector<std::vector<double>> data_type;
    std::ifstream file(std::filesystem::path(path), std::ios::binary);

    if (!file.is_open())
        throw std::invalid_argument("IDX_READER::read_idx: Could not open file");


    // to understand the IDX file format see: http://yann.lecun.com/exdb/mnist/
    char magic[4];
    file.read(magic, 4);

    data_type tensor;

    if (magic[2] == 0x08)
    {
        size_t dimensions = magic[3];
        std::vector<size_t> shape(dimensions);
        for (std::uint32_t i = 0; i < dimensions; i++)
        {
            unsigned char dimension[4];
            file.read((char *)dimension, 4);
            shape[i] = (size_t)dimension[0] << 24 | (size_t)dimension[1] << 16 | (size_t)dimension[2] << 8 | (size_t)dimension[3];
        }
        tensor.resize(shape[0], std::vector<double>(std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<size_t>())));
        for (std::uint32_t i = 0; i < tensor.size(); i++)
        {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            for (std::uint32_t j = 0; j < tensor[i].size(); j++)
            {
                unsigned char pixel;
                file.read((char *)&pixel, 1);
                tensor[i][j] = (double)pixel;
            }
        }
    }
    else
    {
        throw std::invalid_argument("IDX_READER::read_idx: Unknown magic number");
    }

    return tensor;
}


#endif // IDXREADER_HPP