//
// Created by servant-of-scietia on 10/13/24.
//

#ifndef TENSOR_H
#define TENSOR_H

#include "dependencies.h"
#include "config.h"
#include "primitives/enum.h"

namespace brainet
{
	class Tensor
	{
    	std::vector<std::uint64_t> m_dimensions, m_strides;
    	bool m_isVirtual;
    	std::string m_name;
    	dtype_t m_dataType;

  	  public:
    	Tensor(const std::vector<std::uint64_t>  &dimensions, const std::vector<std::uint64_t>  &strides, const dtype_t &dataType, const std::string &name, const bool &isVirtual = false);

    	[[nodiscard]] std::vector<std::uint64_t> getDimensions() const;
    	[[nodiscard]] std::vector<std::uint64_t> getStrides() const;
    	[[nodiscard]] dtype_t getDataType() const;
    	[[nodiscard]] std::string getName() const;
    	[[nodiscard]] bool isVirtual() const;

    	void setDimensions(const std::vector<std::uint64_t> &dimensions);
		void setStrides(const std::vector<std::uint64_t> &strides);
		void setDataType(const dtype_t &dataType);
		void setName(const std::string &name);
		void setIsVirtual(const bool &isVirtual);
	};
} // brainet

#endif //TENSOR_H
