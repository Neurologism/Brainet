//
// Created by servant-of-scietia on 20.09.24.
//

#include "../include/reader.hpp"

Reader::data_type Reader::read_idx(const std::string& path)
{
    std::ifstream file(std::filesystem::path(path), std::ios::binary);

    if (!file.is_open())
        throw std::invalid_argument("IDX_READER::read_idx: Could not open file");

    // to understand the IDX file format, see: http://yann.lecun.com/exdb/mnist/


    auto file_iterator = std::istreambuf_iterator<char>(file);
    auto file_end = std::istreambuf_iterator<char>{};

    std::array<std::byte, 4> magic{};

    for (std::uint32_t i = 0; i < 4; i++)
    {
        magic[i] = static_cast<std::byte>(*file_iterator++);
    }

    Reader::data_type tensor;

    if ( magic[2] == static_cast<std::byte>(0x08) )
    {
        auto dimensions = static_cast<size_t>(magic[3]);
        std::vector<size_t> shape(dimensions);
        for (std::uint32_t i = 0; i < dimensions; i++)
        {
            std::array<std::byte, 4> dimension{};
            for (std::uint32_t j = 0; j < 4; j++)
            {
                dimension[j] = static_cast<std::byte>(*file_iterator++);
            }
            shape[i] = static_cast<size_t>(dimension[0]) << 24 | static_cast<size_t>(dimension[1]) << 16 | static_cast<size_t>(dimension[2]) << 8 | static_cast<size_t>(dimension[3]);
        }

        tensor = Reader::data_type(shape[0], std::vector<Precision>(std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>()), 0));

        for (auto & i : tensor)
        {
            if (file_iterator == file_end)
            {
                throw std::invalid_argument("IDX_READER::read_idx: Unexpected end of file");
            }
            for (Precision & j : i)
            {
                j = static_cast<double>(static_cast<std::byte>(*file_iterator++));
            }
        }
    }
    else
    {
        throw std::invalid_argument("IDX_READER::read_idx: Unknown magic number");
    }

    return tensor;
}

void Reader::read_bin(std::vector<std::vector<Precision>> const & designMatrix, std::vector<std::vector<Precision>> const & label, const std::string &path)
{
    throw std::invalid_argument("Under maintenance");
    // typedef std::vector<std::vector<double>> data_type;
    // std::ifstream file(std::filesystem::path(path), std::ios::binary);
    //
    // if (!file.is_open())
    //     throw std::invalid_argument("BIN_READER::read_bin: Could not open file");
    //
    // // CIFAR-10 format: https://www.cs.toronto.edu/~kriz/cifar.html
    //
    // for (std::uint32_t i = 0; i < 10000; i++)
    // {
    //     std::byte read;
    //     file.read(reinterpret_cast<char*>(&read), sizeof(read));
    //
    //     label.push_back(std::to_integer<double>(read));
    //
    //     designMatrix.push_back(std::vector<double>(3072));
    //
    //     for (std::uint32_t j = 0; j < 3072; j++)
    //     {
    //         std::byte pixel;
    //         file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
    //         designMatrix[i][j] = std::to_integer<double>(pixel);
    //     }
    // }
}