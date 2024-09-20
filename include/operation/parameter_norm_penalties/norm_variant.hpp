//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef NORM_VARIANT_HPP
#define NORM_VARIANT_HPP

#include "L1_Norm.hpp"
#include "L2_Norm.hpp"

using ParameterNormPenaltyVariant = std::variant<L1Norm, L2Norm>;

#endif //NORM_VARIANT_HPP
