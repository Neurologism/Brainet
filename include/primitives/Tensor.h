//
// Created by servant-of-scietia on 10/13/24.
//

#ifndef TENSOR_H
#define TENSOR_H

#include "Dependencies.hpp"
#include "Config.hpp"
#include "DataTypes.h"

class Tensor
{
    std::vector<std::uint64_t> m_dimensions, m_strides;
    bool m_isVirtual;                                         // sets if the data of the tensor can be optimized out
    std::string m_name;
    dtype_t m_dataType;

  public:
    Tensor(std::vector<std::uint64_t> const &dimensions, std::vector<std::uint64_t> const &strides, dtype_t dataType, std::string const &name, bool isVirtual = false);

    std::vector<std::uint64_t> getDimensions() const;
    std::vector<std::uint64_t> getStrides() const;
    dtype_t getDataType() const;
    std::string getName() const;
    bool isVirtual() const;

    void setDimensions(std::vector<std::uint64_t> const &dimensions);
    void setStrides(std::vector<std::uint64_t> const &strides);
    void setDataType(dtype_t dataType);
    void setName(std::string const &name);
    void setVirtual(bool isVirtual);
};

#endif //TENSOR_H
