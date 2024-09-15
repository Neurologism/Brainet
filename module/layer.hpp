//
// Created by servant-of-scietia on 14.09.24.
//

#ifndef LAYER_HPP
#define LAYER_HPP

#include "module.hpp"

/**
  * @brief The Layer class is an extension of the Module class. It provides a framework for creating layers of a neural network.
  */
class Layer : public Module
{
protected:
    explicit Layer(std::string name) : Module(std::move(name)) {}

    std::uint32_t mSize = -1; // stores the number of neurons in the layer

    /**
     * @brief used to get the size of the layer
     */
    [[nodiscard]] std::uint32_t getSize() const;
};

inline std::uint32_t Layer::getSize() const
{
    return mSize;
}

#include "dense.hpp"

using LayerVariant = std::variant<Dense>;

#endif //LAYER_HPP
