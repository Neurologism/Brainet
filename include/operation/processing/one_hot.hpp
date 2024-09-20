#ifndef ONEHOT_HPP
#define ONEHOT_HPP

#include "../operation.hpp"

/**
 * @brief One hot encoding class, used to perform one hot encoding on the input tensor. Only supporting forward pass.
*/
class OneHot : public Operation
{
    std::uint32_t _size; // size of the one hot encoding
    double _on_value;
    double _off_value;

public:
    /**
     * @brief One hot encoding constructor.
     * @param size The size of the one hot encoding.
     * @param on_value The value to set the one hot encoding to.
     * @param off_value The value to set the rest of the tensor to.
     */
    OneHot(std::uint32_t size, double on_value, double off_value);
    ~OneHot() = default;

    /**
     * @brief Perform one hot encoding on the input tensor.
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs)override;

    /**
     * @brief Backward pass is not supported for one hot encoding.
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)override;
};

#endif // ONEHOT_HPP
    