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

    inline std::shared_ptr<Operation> createOperation(JsonNode & node)
    {
        if (!node.m_children.contains("value"))
        {
            invalidJson();
        }
        if (std::get<std::string>(node.m_children["value"]) == "relu")
        {
            return std::make_shared<ReLU>();
        }
        if (std::get<std::string>(node.m_children["value"]) == "softmax")
        {
            return std::make_shared<Softmax>();
        }
        if (std::get<std::string>(node.m_children["value"]) == "linear")
        {
            return std::make_shared<Linear>();
        }
        invalidJson();
        return nullptr;
    }

    inline LossFunctionVariant createLoss(JsonNode & node)
    {
        if (!node.m_children.contains("value"))
        {
            invalidJson();
        }
        if (std::get<std::string>(node.m_children["value"]) == "error_rate")
        {
            return ErrorRate();
        }
        invalidJson();
        throw std::runtime_error("This should never be reached.");
    }

    inline OptimizerVariant createOptimizer(JsonNode & node)
    {
        if (!node.m_children.contains("value"))
        {
            invalidJson();
        }
        if (std::get<std::string>(node.m_children["value"]) == "sgd")
        {
            auto params = std::get<std::vector<JsonNode>>(node.m_children["parameters"]);
            return SGD(std::get<double>(params[0].m_children["value"]), std::get<int>(params[1].m_children["value"]));
        }
        invalidJson();
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

        if (!m_root.m_children.contains("operations"))
        {
            invalidJson();
        }
        auto child = std::get<JsonNode>(m_root.m_children["operations"]);

        if (!child.m_children.contains("add") || !child.m_children.contains("train") || !child.m_children.contains("predict"))
        {
            invalidJson();
        }
        auto modules = std::get<std::vector<JsonNode>>(std::get<JsonNode>(child.m_children["add"]).m_children["modules"]);
        auto connections = std::get<std::vector<JsonNode>>(std::get<JsonNode>(child.m_children["add"]).m_children["connections"]);

        for (auto & module : modules)
        {
            if (module.m_children.contains("value"))
            {
                invalidJson();
            }
            auto params = std::get<std::vector<JsonNode>>(module.m_children["parameters"]);
            if (std::get<std::string>(module.m_children["type"]) == "dense")
            {
                model.addModule(Dense(createOperation(params[0]), std::get<int>(params[1].m_children["value"]),
                    std::get<std::string>(params[2].m_children["value"])));
            }
            else if (std::get<std::string>(module.m_children["type"]) == "loss")
            {
                model.addModule(Loss(createLoss(params[0]), std::get<std::string>(params[1].m_children["value"])));
            }
            else
            {
                invalidJson();
            }
        }

        for (auto & connection : connections)
        {
            if (!connection.m_children.contains("from") || !connection.m_children.contains("to"))
            {
                invalidJson();
            }
            model.connectModules(std::get<std::string>(connection.m_children["from"]), std::get<std::string>(connection.m_children["to"]));
        }
        auto trainingParams = std::get<std::vector<JsonNode>>(std::get<JsonNode>(child.m_children["train"]).m_children["parameters"]);

        // mnist dataset is only dataset supported
        typedef std::vector<std::vector<double>> dataType;
        dataType train_input = read_idx("../mnist/train-images.idx3-ubyte");
        dataType train_target = read_idx("../mnist/train-labels.idx1-ubyte");

        dataType test_input = read_idx("../mnist/t10k-images.idx3-ubyte");
        dataType test_target = read_idx("../mnist/t10k-labels.idx1-ubyte");

        train_input = preprocessing::normalize(train_input);
        test_input = preprocessing::normalize(test_input);

        Dataset mnist(train_input, train_target, 0.99, test_input, test_target);


        model.train(mnist, std::get<std::string>(trainingParams[0].m_children["value"]), std::get<std::string>(trainingParams[1].m_children["value"]),
            std::get<int>(trainingParams[2].m_children["value"]), std::get<int>(trainingParams[3].m_children["value"]),
            createOptimizer(trainingParams[4]), std::get<int>(trainingParams[5].m_children["value"]));



        auto testParams = std::get<std::vector<JsonNode>>(std::get<JsonNode>(child.m_children["predict"]).m_children["parameters"]);

        model.test(mnist, std::get<std::string>(testParams[0].m_children["value"]), std::get<std::string>(testParams[1].m_children["value"]));

    }
}


#endif //JSON_BRAINET_BRIDGE_HPP
