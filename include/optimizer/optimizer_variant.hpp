//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef OPTIMIZER_VARIANT_HPP
#define OPTIMIZER_VARIANT_HPP

#include "adagrad.hpp"
#include "adam.hpp"
#include "momentum_sgd.hpp"
#include "nesterov_momentum.hpp"
#include "rmsprop.hpp"
#include "rmsprop_nesterov.hpp"
#include "sgd.hpp"

using OptimizerVariant = std::variant<AdaGrad, Adam, Momentum, NesterovMomentum, RMSProp, RMSPropNesterov, SGD>;

#endif //OPTIMIZER_VARIANT_HPP
