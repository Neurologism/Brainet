//
// Created by servant-of-scietia on 11.09.24.
//
#pragma GCC optimize("O3")
#include "json_brainet_bridge.hpp"

int main(int argc, char *argv[])
{
    Logger::msJsonFormat = true;
    JSON::convertJsonToBrainet("../json_interface/task.json");
}
