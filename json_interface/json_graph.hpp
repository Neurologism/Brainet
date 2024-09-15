//
// Created by servant-of-scietia on 10.09.24.
//

#ifndef JSON_GRAPH_HPP
#define JSON_GRAPH_HPP

#include "../dependencies.hpp"

namespace JSON
{
    struct JsonNode;
    typedef std::variant<bool, int, double, std::string, std::vector<JsonNode>, JsonNode> JsonType;

    struct JsonNode
    {
        std::map<std::string, JsonType> m_children;
    };

    struct JsonGraph
    {
        JsonNode m_root;
    };
}
#endif //JSON_GRAPH_HPP
