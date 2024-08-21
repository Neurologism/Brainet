#ifndef BIN_READER_HPP
#define BIN_READER_HPP

#include "../dependencies.hpp"

/**
 * @brief This function reads a binary file in the CIFAR-10 format and returns the data as a 2D vector.
 * @param designMatrix The design matrix of the data.
 * @param label The label of the data.
 * @param path The path to the binary file.
 */
void read_bin(std::vector<std::vector<double>> const & designMatrix, std::vector<std::vector<double>> const & label, const std::string path)
{
    typedef std::vector<std::vector<double>> data_type;
    std::ifstream file(std::filesystem::path(path), std::ios::binary);

    if (!file.is_open())
        throw std::invalid_argument("BIN_READER::read_bin: Could not open file");

    // CIFAR-10 format: https://www.cs.toronto.edu/~kriz/cifar.html

    for (std::uint32_t i = 0; i < 10000; i++)
    {
        std::byte label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        label.push_back(std::to_integer<double>(label));

        designMatrix.push_back(std::vector<double>(3072));

        for (std::uint32_t j = 0; j < 3072; j++)
        {
            std::byte pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            designMatrix[i][j] = std::to_integer<double>(pixel);
        }
    }
}



    