#ifndef IDXREADER_HPP
#define IDXREADER_HPP

#include "../dependencies.hpp"

/**
 * @brief This function reads an IDX file and returns the data as a 2D vector.
 * @param path The path to the IDX file.
 * @return The data as a 2D vector.
 */
inline std::vector<std::vector<double>> read_idx(const std::string& path)
{
    typedef std::vector<std::vector<double>> data_type;
    std::ifstream file(std::filesystem::path(path), std::ios::binary);

    if (!file.is_open())
        throw std::invalid_argument("IDX_READER::read_idx: Could not open file");
    
    // to understand the IDX file format see: http://yann.lecun.com/exdb/mnist/


    auto file_iterator = std::istreambuf_iterator<char>(file);
	auto file_end = std::istreambuf_iterator<char>{};

    std::array<std::byte, 4> magic{};

    for (std::uint32_t i = 0; i < 4; i++)
    {
        magic[i] = static_cast<std::byte>(*file_iterator++);
    }

    data_type tensor;

    if ( magic[2] == std::byte(0x08) )
    {
        size_t dimensions = (size_t)magic[3];
        std::vector<size_t> shape(dimensions);
        for (std::uint32_t i = 0; i < dimensions; i++)
        {
            std::array<std::byte, 4> dimension;
            for (std::uint32_t j = 0; j < 4; j++)
            {
                dimension[j] = std::byte(*file_iterator++);
            }
            shape[i] = static_cast<size_t>(dimension[0]) << 24 | static_cast<size_t>(dimension[1]) << 16 | static_cast<size_t>(dimension[2]) << 8 | static_cast<size_t>(dimension[3]);
        }

        tensor = data_type(shape[0], std::vector<double>(std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<size_t>()), 0));

        for (std::uint32_t i = 0; i < tensor.size(); i++)
        {
            if (file_iterator == file_end)
            {
                throw std::invalid_argument("IDX_READER::read_idx: Unexpected end of file");
            }
            for (std::uint32_t j = 0; j < tensor[i].size(); j++)
            {
                tensor[i][j] = static_cast<size_t>(std::byte(*file_iterator++));
                
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