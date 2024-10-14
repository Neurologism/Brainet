//
// Created by servant-of-scietia on 10/13/24.
//

#include "Tensor.h"

namespace brainet
{
    Tensor::Tensor(const std::vector<std::uint64_t>& dimensions, const std::vector<std::uint64_t>& strides, const dtype_t& dataType, const std::string& name, const bool& isVirtual)
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

    void Tensor::setDimensions(const std::vector<std::uint64_t>& dimensions)
    {
        m_dimensions = dimensions;
    }

    void Tensor::setStrides(const std::vector<std::uint64_t>& strides)
    {
        m_strides = strides;
    }

    void Tensor::setDataType(const dtype_t& dataType)
    {
        m_dataType = dataType;
    }

    void Tensor::setName(const std::string& name)
    {
        m_name = name;
    }

    void Tensor::setIsVirtual(const bool& isVirtual)
    {
        m_isVirtual = isVirtual;
    }

} // brainet