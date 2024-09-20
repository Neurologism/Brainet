#ifndef PADDING_HPP
#define PADDING_HPP

#include "../operation.hpp"

/**
 * @brief Padding class, used to add padding in positive x and y direction to the input tensor in shape (x, y). Should be extended to work for any amount of dimensions and in both directions.
*/
class Padding : public Operation
{
    // store data
    std::uint32_t _x_padding;
    std::uint32_t _y_padding;
    double _padding_value;

public:
    /**
     * @brief Add a padding unit to the graph.
     * @param x_padding The amount of padding in the x direction.
     * @param y_padding The amount of padding in the y direction.
     * @param padding_value The value of the padding. 
     */
    Padding(std::uint32_t x_padding, std::uint32_t y_padding, double padding_value);
    ~Padding() = default;

    /**
     * @brief Add padding to the input tensor.
     */
    virtual void f(std::vector<std::shared_ptr<Variable>>& inputs)override;
    /**
     * @brief Remove padding from the gradient tensor.
     */
    virtual std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)override;
};

#endif // PADDING_HPP