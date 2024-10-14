//
// Created by servant-of-scietia on 10/14/24.
//

#include "Node.h"

#include <utility>

namespace brainet
{
    Node::Node(std::string name) : m_name(std::move(name))
    {
    }

    std::vector<std::shared_ptr<Tensor>> Node::getTensorIn() const
    {
        return m_tensorIn;
    }

    std::vector<std::shared_ptr<Tensor>> Node::getTensorOut() const
    {
        return m_tensorOut;
    }

    std::string Node::getName() const
    {
        return m_name;
    }

    void Node::setTensorIn(const std::vector<std::shared_ptr<Tensor>> &tensorIn)
    {
        m_tensorIn = tensorIn;
    }

    void Node::setTensorOut(const std::vector<std::shared_ptr<Tensor>> &tensorOut)
    {
        m_tensorOut = tensorOut;
    }

    void Node::setName(const std::string &name)
    {
        m_name = name;
    }
} // brainet