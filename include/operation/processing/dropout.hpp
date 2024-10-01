#ifndef DROP_OUT_HPP
#define DROP_OUT_HPP

#include "../operation.hpp"

/**
 * @brief Dropout class, representing the dropout operation.
*/
class Dropout : public Operation
{
    double mDropoutRate;
    std::vector<bool> mMask;
    static bool msAveraging; // indicates if the dropout is in training or testing mode

public:
    Dropout(double dropoutRate) : mDropoutRate(dropoutRate) { mName = "DROPOUT"; }
    void f(std::vector<std::shared_ptr<Variable>>& inputs) override;
    std::shared_ptr<Tensor> bprop(std::vector<std::shared_ptr<Variable>>& inputs, std::shared_ptr<Variable> & focus, std::shared_ptr<Tensor> & gradient) override;
    static void activateAveraging() { msAveraging = true; }
    static void deactivateAveraging() { msAveraging = false; }
};

#endif // DROP_OUT_HPP