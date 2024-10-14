//
// Created by servant-of-scietia on 9/28/24.
//

#ifndef CONFIG_H
#define CONFIG_H

#define DOUBLE_PRECISION 0

#if DOUBLE_PRECISION
    using Precision = double;
#else
    using Precision = float;
#endif

#endif //CONFIG_H
