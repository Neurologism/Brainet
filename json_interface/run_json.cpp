//
// Created by servant-of-scietia on 11.09.24.
//
#pragma GCC optimize("O3")
#include "json_brainet_bridge.hpp"

int main(int argc, char *argv[])
{
    std::ios::sync_with_stdio(false);
    JSON::convertJsonToBrainet("../json_interface/task.json");
}
