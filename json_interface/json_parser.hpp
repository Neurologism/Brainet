//
// Created by servant-of-scietia on 10.09.24.
//

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include "../dependencies.hpp"
#include "json_graph.hpp"

namespace JSON
{
    /**
     * @brief This function performs a depth first search on the json graph.
     * @param node The node to start the search from.
     */
    void depthFirstSearch(JsonNode * node, std::ifstream & file)
    {
        char ch = '{';
        while (ch != '}') // end of the json object
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

                while (ch != '}') // end of the json object
                {
                    file.get(ch);
                    switch (ch)
                    {
                        case '{':
                        {
                            auto child = std::make_unique<JsonNode>();
                            depthFirstSearch(child.get(), file);
                            node->m_children.emplace_back(key, std::move(child));
                            break;
                        }
                        case '[':
                        {
                            std::vector<std::unique_ptr<JsonNode>> children;
                            while (ch != ']')
                            {
                                auto child = std::make_unique<JsonNode>();
                                depthFirstSearch(child.get(), file);
                                children.push_back(std::move(child));
                            }
                            node->m_children.emplace_back(key, children);
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
                            node->m_children.emplace_back(key, value);
                            break;
                        }
                        case 't':
                        {
                            node->m_children.emplace_back(key, true);
                            break;
                        }
                        case 'f':
                        {
                            node->m_children.emplace_back(key, false);
                            break;
                        }
                        case 'n':
                        {
                            node->m_children.push_back(std::make_pair(key, nullptr));
                            break;
                        }
                        default:
                        {
                            if (std::isdigit(ch))
                            {
                                std::string value;
                                value.push_back(ch);
                                while (std::isdigit(ch))
                                {
                                    value.push_back(ch);
                                    file.get(ch);
                                }
                                node->m_children.emplace_back(key, std::stoi(value));
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

        auto root = std::make_unique<JsonNode>();
        depthFirstSearch(root.get(), file);
        graph.m_root = std::move(root);


        return graph;
    }
}

#endif //JSON_PARSER_HPP
