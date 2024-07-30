#ifndef IDX_READER_INCLUDE_GUARD
#define IDX_READER_INCLUDE_GUARD

#include "../dependencies.h"
#include "../tensor.h"

TENSOR<double> read_idx(std::string path)
{
   std::ifstream file(std::filesystem::path(path), std::ios::binary);

    if (!file.is_open())
        throw std::invalid_argument("IDX_READER::read_idx: Could not open file");

    char magic[4];
    file.read(magic, 4);

    TENSOR<double> tensor;

    if (magic[2] == 0x08)
    {
        int dimensions = magic[3];
        std::vector<int> shape(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            unsigned char dimension[4];
            file.read((char *)dimension, 4);
            shape[i] = (unsigned int)dimension[0] << 24 | (unsigned int)dimension[1] << 16 | (unsigned int)dimension[2] << 8 | (unsigned int)dimension[3];
        }
        tensor = TENSOR<double>(shape);
        for (int i = 0; i < tensor.size(); i++)
        {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            tensor.data()[i] = (double)pixel;
        }
    }
    else
    {
        throw std::invalid_argument("IDX_READER::read_idx: Unknown magic number");
    }

    return tensor;
}


#endif // IDX_READER_INCLUDE_GUARD