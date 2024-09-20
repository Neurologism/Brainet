//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef MODULE_VARIANT_HPP
#define MODULE_VARIANT_HPP

#include "dataset.hpp"
#include "loss.hpp"
#include "dense.hpp"

using ModuleVariant = std::variant<Dense, Loss, Dataset>;


using LayerVariant = std::variant<Dense>;

#endif //MODULE_VARIANT_HPP
