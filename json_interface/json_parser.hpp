//
// Created by servant-of-scietia on 10.09.24.
//

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include "dependencies.hpp"
#include "json_graph.hpp"

namespace JSON
{
    /**
     * @brief This function performs a depth-first search on the JSON graph.
     * @param node The node to start the search from.
     * @param file The file stream.
     */
    inline void depthFirstSearch(JsonNode &node, std::ifstream & file) //NOLINT
    {
        char ch = '{';
        while (ch != '}') // end of the JSON object
        {
            file.get(ch);
            if (ch == '"')
            {
                std::string key;
                file.get(ch);
                while (ch != '"')
                {
                    key.push_back(ch);
                    file.get(ch);
                }

                while (ch != ':')
                {
                    file.get(ch);
                }

                bool foundValue = false;
                while (!foundValue)
                {
                    file.get(ch);
                    switch (ch)
                    {
                        case '{':
                        {
                            node.m_children[key] = JsonNode();
                            depthFirstSearch(std::get<JsonNode>(node.m_children[key]), file);
                            foundValue = true;
                            break;
                        }
                        case '[':
                        {
                            node.m_children[key] = std::vector<JsonNode>();
                            while (ch != ']')
                            {
                                file.get(ch);
                                if (ch != '{')
                                {
                                    continue;
                                }
                                std::get<std::vector<JsonNode>>(node.m_children[key]).emplace_back();
                                depthFirstSearch(std::get<std::vector<JsonNode>>(node.m_children[key]).back(), file);
                            }
                            foundValue = true;
                            break;
                        }
                        case '"':
                        {
                            std::string value;
                            file.get(ch);
                            while (ch != '"')
                            {
                                value.push_back(ch);
                                file.get(ch);
                            }
                            node.m_children[key] = value;
                            foundValue = true;
                            break;
                        }
                        case 't':
                        {
                            node.m_children[key] = true;
                            foundValue = true;
                            break;
                        }
                        case 'f':
                        {
                            node.m_children[key] = false;
                            foundValue = true;
                            break;
                        }
                        default:
                        {
                            if (std::isdigit(ch))
                            {
                                std::string value;
                                bool isDouble = false;
                                while (std::isdigit(ch) || ch == '.')
                                {
                                    if (ch == '.')
                                    {
                                        isDouble = true;
                                    }
                                    value.push_back(ch);
                                    file.get(ch);
                                }
                                if (isDouble)
                                {
                                    node.m_children[key] = std::stod(value);
                                }
                                else
                                {
                                    node.m_children[key] = std::stoi(value);
                                }
                                foundValue = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    inline JsonGraph parseJson(const std::string & json)
    {
        JsonGraph graph;
        std::ifstream file{std::filesystem::path(json)};

        if ( !file.is_open())
            throw std::invalid_argument("JSON_PARSER::parseJson: Could not open file");


        char ch;
        file.get(ch);
        while (ch != '{')
        {
            file.get(ch);
        }

        depthFirstSearch(graph.m_root, file);

        return graph;
    }
}

#endif //JSON_PARSER_HPP
