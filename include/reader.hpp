//
// Created by servant-of-scietia on 20.09.24.
//

#ifndef READER_HPP
#define READER_HPP

#include <datatypes/tensor.hpp>

#include "dependencies.hpp"

class Reader
{
public:
    typedef std::vector<std::vector<Precision>> data_type;
    static void read_bin(data_type const & designMatrix, data_type const & label, const std::string &path);
    static data_type read_idx(const std::string& path);
};

#endif //READER_HPP
