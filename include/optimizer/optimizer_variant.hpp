//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef OPTIMIZER_VARIANT_HPP
#define OPTIMIZER_VARIANT_HPP

#include "adam.hpp"
#include "sgd.hpp"

using OptimizerVariant = std::variant<SGD>;

#endif //OPTIMIZER_VARIANT_HPP
