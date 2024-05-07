#ifndef ABS_VARIABLE_INCLUDE_GUARD
#define ABS_VARIABLE_INCLUDE_GUARD

#include "dependencies.h"

/**
 * @brief Abstract class for the variable class
*/
class ABS_VARIABLE
{
    virtual std::vector<double> get_data();
    virtual std::vector<int> get_shape();
    virtual int get_id();
    virtual void set_data(std::vector<double> & data);
    virtual void set_shape(std::vector<int> & shape);
    virtual void set_id(int id);
};

#endif // ABS_VARIABLE_INCLUDE_GUARD