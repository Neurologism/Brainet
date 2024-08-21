#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "../operation.hpp"

/**
 * @brief The CrossEntropy class is a cost function that is used to train a model using the negative log likelyhood.
 */
class CrossEntropy : public Operation
{
    bool mUseWithLog = true;
public:
    /**
     * @brief constructor for the CrossEntropy operation
     */
    CrossEntropy() { mName = "NEGATIVE_LOG_LIKELYHOOD"; };
    ~CrossEntropy() = default;

    /**
     * @brief calculate the negative log likelyhood of the input tensors
     * @param inputs The input tensors
     * @note The first input tensor is the prediction and the second input tensor is the target.
     */
    void f(std::vector<std::shared_ptr<Variable>> &inputs) override;

    /**
     * @brief calculate the gradient of the negative log likelyhood
     * @param inputs The input tensors
     * @param focus The focus tensor
     * @param gradient The gradient tensor
     * @return The gradient tensor
     */
    std::shared_ptr<Tensor<double>> bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient) override;

    void useWithExp();
};

void CrossEntropy::f(std::vector<std::shared_ptr<Variable>> &inputs)
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("CrossEntropy: number of inputs is not 2");
    }

    if (inputs[0]->getData()->shape(1) != 1)
    {
        throw std::runtime_error("CrossEntropy: the target tensor must be 1D");
    }

    if (inputs[1]->getData()->shape(0) != inputs[0]->getData()->shape(0))
    {
        throw std::runtime_error("CrossEntropy: the size of the prediction and target tensor must be the same");
    }

    double error = 0;
    for (std::uint32_t i = 0; i < inputs[1]->getData()->shape(0); i++)
    {     
        if (mUseWithLog)
        {
            error -= log(inputs[1]->getData()->at({i, (std::uint32_t)inputs[0]->getData()->at({i})}));
        }
        else
        {
            error -= inputs[1]->getData()->at({i, (std::uint32_t)inputs[0]->getData()->at({i})});
        }
    }
    // std::vector<std::uint32_t> prediction(10);
    // std::vector<std::uint32_t> target(10);

    // for (std::uint32_t i = 0; i < inputs[1]->getData()->shape(0); i++)
    // {
    //     double max = inputs[1]->getData()->at({i, 0});
    //     std::uint32_t maxIndex = 0;
    //     for (std::uint32_t j = 1; j < inputs[1]->getData()->shape(1); j++)
    //     {
    //         if (inputs[1]->getData()->at({i, j}) > max)
    //         {
    //             max = inputs[1]->getData()->at({i, j});
    //             maxIndex = j;
    //         }
    //     }
    //     prediction[maxIndex]++;
    //     target[(std::uint32_t)inputs[0]->getData()->at({i})]++;
    // }

    // for (std::uint32_t i = 0; i < 10; i++)
    // {
    //     std::cout << "Prediction: " << prediction[i] << " Target: " << target[i] << std::endl;
    // }

    this->getVariable()->getData() = std::make_shared<Tensor<double>>(Tensor<double>({1}, error / inputs[1]->getData()->shape(0)));
}


std::shared_ptr<Tensor<double>> CrossEntropy::bprop(std::vector<std::shared_ptr<Variable>> &inputs, std::shared_ptr<Variable> &focus, std::shared_ptr<Tensor<double>> &gradient)
{
    if (inputs.size() != 2)
    {
        throw std::runtime_error("CrossEntropy: number of inputs is not 2");
    }

    if (inputs[0]->getData()->shape(1) != 1)
    {
        throw std::runtime_error("CrossEntropy: the target tensor must be 1D");
    }

    if (gradient->shape() != std::vector<size_t>({1}))
    {
        throw std::runtime_error("CrossEntropy: the gradient tensor must have shape {1}");
    }

    std::shared_ptr<Tensor<double>> _gradient = std::make_shared<Tensor<double>>(Tensor<double>(inputs[1]->getData()->shape()));

    for (std::uint32_t i = 0; i < inputs[1]->getData()->shape(0); i++)
    {
        for (std::uint32_t j = 0; j < inputs[1]->getData()->shape(1); j++)
        {
            if (mUseWithLog)
            {
                _gradient->set({i, j}, -1 / inputs[1]->getData()->at({i, j}) * (j == (std::uint32_t)inputs[0]->getData()->at({i})) / inputs[1]->getData()->capacity());
            }
            else
            {
                _gradient->set({i, j}, -1*(j == (std::uint32_t)inputs[0]->getData()->at({i})) / inputs[1]->getData()->capacity());
            }
        }
    }

    return _gradient;
}

void CrossEntropy::useWithExp()
{
    mUseWithLog = false;
}


#endif // CROSS_ENTROPY_HPP