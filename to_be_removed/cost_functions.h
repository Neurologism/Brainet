#ifndef COST_FUNCTIONS_INCLUDE_GUARD
#define COST_FUNCTIONS_INCLUDE_GUARD


#include<vector>
#include<stdexcept>

namespace cost_functions
{
    double mean_squared_error(std::vector<double> prediction, std::vector<double> target)
    {
        if(prediction.size() != target.size())
        {
            throw std::invalid_argument("Dimensionality of prediction vector and target vector do not match.");
        }
        
    }
} // namespace activation_functions




#endif