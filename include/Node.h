//
// Created by servant-of-scietia on 10/14/24.
//

#ifndef NODE_H
#define NODE_H

#include <utility>

#include "Tensor.h"

namespace brainet
{
    class Node
    {
        std::vector<std::shared_ptr<Tensor>> m_tensorIn, m_tensorOut;
        std::string m_name;

      public:
        explicit Node(std::string name = "Node");

        [[nodiscard]] std::vector<std::shared_ptr<Tensor>> getTensorIn() const;
        [[nodiscard]] std::vector<std::shared_ptr<Tensor>> getTensorOut() const;
        [[nodiscard]] std::string getName() const;

        void setTensorIn(const std::vector<std::shared_ptr<Tensor>> &tensorIn);
        void setTensorOut(const std::vector<std::shared_ptr<Tensor>> &tensorOut);
        void setName(const std::string &name);
    };
} // brainet

#endif //NODE_H
