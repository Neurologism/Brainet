//
// Created by servant-of-scietia on 11.09.24.
//

#ifndef JSON_BRAINET_BRIDGE_HPP
#define JSON_BRAINET_BRIDGE_HPP

#include "json_parser.hpp"
#include "../brainet.hpp"

namespace JSON
{

    inline void invalidJson()
    {
        throw std::runtime_error("The format of the json file is invalid.");
    }
    /**
     * @brief This function parses the json file and runs the brainet model accordingly.
     * @param file The file to parse.
     */
    inline void convertJsonToBrainet(const std::string & file)
    {
        auto [m_root] = parseJson(file);

        if (m_root->m_children[0].first != "operations")
        {
            invalidJson();
        }
        auto children = std::get<std::vector<std::unique_ptr<JsonNode>>>(m_root->m_children[0].second);

        for (auto & child : children)
        {
            if (child->m_children[0].first != "type")
            {
                invalidJson();
            }
            switch (std::get<std::string>(child->m_children[0].second))
            {
                case "add":
                {

                }


                case "train":
                {

                }


                case "test":
                {

                }
            }
        }
    }
}


#endif //JSON_BRAINET_BRIDGE_HPP
