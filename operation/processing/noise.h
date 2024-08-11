#ifndef NOISE_INCLUDE_GUARD
#define NOISE_INCLUDE_GUARD

#include "../operation.h"

/**
 * @brief Used to add random noise to an input tensor. This is used to augment the dataset.
 */
class Noise : public Operation
{
    double _mean;
    double _stddev;

public:

    /**
     * @brief add random noise to the input tensor
     * @param mean the mean of the noise
     * @param stddev the standard deviation of the noise
     */
    Noise(double mean, double stddev) : _mean(mean), _stddev(stddev) { __dbg_name = "NOISE"; };
    ~Noise() = default;

    /**
     * @brief add random noise to the input tensor
     */
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;

    /**
     * @brief backward pass is not supported for noise
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient) override;
};

void Noise::f(std::vector<std::shared_ptr<Variable>>& inputs)
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Noise: number of inputs is not 1");
    }

    auto input = inputs[0]->get_data();
    auto result = std::make_shared<Tensor<double>>(Tensor<double>(input->shape()));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(_mean, _stddev);

    for (std::uint32_t i = 0; i < input->size(); i++)
    {
        result->set({i}, input->at({i}) + dist(gen));
    }

    this->get_variable()->get_data() = result;
}

std::shared_ptr<Tensor<double>> Noise::bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor<double>> & gradient)
{
    throw std::runtime_error("Noise: backward pass is not supported");
}

#endif // NOISE_INCLUDE_GUARD