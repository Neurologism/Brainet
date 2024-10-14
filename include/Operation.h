//
// Created by servant-of-scietia on 10/14/24.
//

#ifndef OPERATION_H
#define OPERATION_H

#include "Node.h"

namespace brainet
{
    class Operation : public Node
    {
      public:
        explicit Operation(const std::string &name = "Operation");
    }
} // brainet



#endif //OPERATION_H
