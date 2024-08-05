#ifndef NORM_INCLUDE_GUARD
#define NORM_INCLUDE_GUARD

#include "L1.h"
#include "L2.h"

using NORM_VARIANT = std::variant<L1_NORM, L2_NORM>;


#endif // NORM_INCLUDE_GUARD