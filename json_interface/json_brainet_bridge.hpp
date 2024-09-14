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

    inline std::shared_ptr<Operation> createOperation(const std::unique_ptr<JsonNode> & node)
    {
        if (node->m_children[0].first != "type")
        {
            invalidJson();
        }
        switch (std::get<std::string>(node->m_children[0].second))
        {
            case "relu":
            {
                if (node->m_children.size() == 2)
                {
                    return std::make_shared<ReLU>(std::get<double>(node->m_children[1].second));
                }
                return std::make_shared<ReLU>();
            }
            case "softmax":
            {
                return std::make_shared<Softmax>();
            }
            case "linear":
            {
                return std::make_shared<Linear>();
            }
            default:
            {
                invalidJson();
            }
        }
        return nullptr;
    }

    inline LossFunctionVariant createLoss(const std::unique_ptr<JsonNode> & node)
    {
        if (node->m_children[0].first != "type")
        {
            invalidJson();
        }
        switch (std::get<std::string>(node->m_children[0].second))
        {
            case "error_rate":
            {
                return ErrorRate();
            }
            default:
            {
                invalidJson();
            }
        }
        throw std::runtime_error("This should never be reached.");
    }

    inline OptimizerVariant createOptimizer(const std::unique_ptr<JsonNode> & node)
    {
        if (node->m_children[0].first != "type")
        {
            invalidJson();
        }
        switch (std::get<std::string>(node->m_children[0].second))
        {
            case "sgd":
            {
                return SGD(std::get<double>(node->m_children[1].second), std::get<std::uint32_t>(node->m_children[2].second));
            }
            default:
            {
                invalidJson();
            }
        }
        throw std::runtime_error("This should never be reached.");
    }


    /**
     * @brief This function parses the JSON file and runs the brainet model accordingly.
     * @param file The file to parse.
     */
    inline void convertJsonToBrainet(const std::string & file)
    {
        Model model;

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
                    for (const auto & grandChild : std::get<std::vector<std::unique_ptr<JsonNode>>>(child->m_children[1].second))
                    {
                        if (grandChild->m_children[0].first != "type")
                        {
                            invalidJson();
                        }
                        auto params = std::get<std::vector<std::unique_ptr<JsonNode>>>(grandChild->m_children[1].second);
                        switch (std::get<std::string>(grandChild->m_children[0].second))
                        {
                            case "dense":
                            {
                                model.addModule(Dense(createOperation(params[0]), std::get<std::uint32_t>(params[1]->m_children[1].second), std::get<std::string>(params[2]->m_children[1].second)));
                            }
                            case "loss":
                            {
                                model.addModule(Loss(createLoss(params[2]), std::get<std::string>(params[3]->m_children[1].second)));
                            }
                            default:
                            {
                                invalidJson();
                            }
                        }
                    }

                    for (const auto & grandChild : std::get<std::vector<std::unique_ptr<JsonNode>>>(child->m_children[2].second))
                    {
                        if (grandChild->m_children[0].first != "from" || grandChild->m_children[1].first != "to")
                        {
                            invalidJson();
                        }
                        model.connectModules(std::get<std::string>(grandChild->m_children[0].second), std::get<std::string>(grandChild->m_children[1].second));
                    }
                }


                case "train":
                {
                    typedef std::vector<std::vector<double>> dataType;
                    const dataType train_input = read_idx("../mnist/train-images.idx3-ubyte");
                    const dataType train_target = read_idx("../mnist/train-labels.idx1-ubyte");

                    dataType test_input = read_idx("../mnist/t10k-images.idx3-ubyte");
                    dataType test_target = read_idx("../mnist/t10k-labels.idx1-ubyte");
                    model.train(Dataset(train_input, train_target, 0.998, test_input, test_target), std::get<std::string>(child->m_children[1].second), std::get<std::string>(child->m_children[2].second), std::get<std::uint32_t>(child->m_children[3].second), std::get<std::uint32_t>(child->m_children[4].second), createOptimizer(std::get<std::unique_ptr<JsonNode>>(child->m_children[5].second)), std::get<std::uint32_t>(child->m_children[6].second));
                }


                case "test":
                {
                    typedef std::vector<std::vector<double>> dataType;
                    const dataType train_input = read_idx("../mnist/train-images.idx3-ubyte");
                    const dataType train_target = read_idx("../mnist/train-labels.idx1-ubyte");

                    dataType test_input = read_idx("../mnist/t10k-images.idx3-ubyte");
                    dataType test_target = read_idx("../mnist/t10k-labels.idx1-ubyte");
                    model.test(Dataset(train_input, train_target, 0.998, test_input, test_target), std::get<std::string>(child->m_children[1].second), std::get<std::string>(child->m_children[2].second));
                }

                default:
                {
                    invalidJson();
                }
            }
        }
    }
}


#endif //JSON_BRAINET_BRIDGE_HPP
