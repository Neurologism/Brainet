//
// Created by servant-of-scietia on 10/13/24.
//

#include "primitives/Tensor.h"

Tensor::Tensor(std::vector<std::uint64_t> const &dimensions, std::vector<std::uint64_t> const &strides, dtype_t dataType, std::string const &name, bool isVirtual)
    : m_dimensions(dimensions), m_strides(strides), m_dataType(dataType), m_name(name), m_isVirtual(isVirtual)
{
}

std::vector<std::uint64_t> Tensor::getDimensions() const
{
    return m_dimensions;
}

std::vector<std::uint64_t> Tensor::getStrides() const
{
    return m_strides;
}

dtype_t Tensor::getDataType() const
{
    return m_dataType;
}

std::string Tensor::getName() const
{
    return m_name;
}

bool Tensor::isVirtual() const
{
    return m_isVirtual;
}

void Tensor::setDimensions(std::vector<std::uint64_t> const &dimensions)
{
    m_dimensions = dimensions;
}

void Tensor::setStrides(std::vector<std::uint64_t> const &strides)
{
    m_strides = strides;
}

void Tensor::setDataType(dtype_t dataType)
{
    m_dataType = dataType;
}

void Tensor::setName(std::string const &name)
{
    m_name = name;
}

void Tensor::setVirtual(bool isVirtual)
{
    m_isVirtual = isVirtual;
}