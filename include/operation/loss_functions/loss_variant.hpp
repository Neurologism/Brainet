//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef LOSS_VARIANT_HPP
#define LOSS_VARIANT_HPP

// loss function variant
#include "error_rate.hpp"

using LossFunctionVariant = std::variant<ErrorRate>;

#endif //LOSS_VARIANT_HPP
