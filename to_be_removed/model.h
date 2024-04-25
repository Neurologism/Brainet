#ifndef TRAIN_INCLUDE_GUARD
#define TRAIN_INCLUDE_GUARD

#include<rectified_linear_unit.h>

/**
 * @attention This is just a rough sketch to try basic functionality
*/
class MODEL
{
    std::vector<LAYER> __model;
protected:
    std::vector<double> forward_pass(std::vector<double>data, double target);
    void precalculate_gradient();
    std::vector<double> backward_pass();
public:
    MODEL(){};
    void add(LAYER layer);
};

std::vector<double> MODEL::forward_pass(std::vector<double>data, double target)
{
    for(LAYER & _layer : __model)
    {
        data = _layer.operation(data);
    }
    return data;
}

void MODEL::precalculate_gradient()
{
    for(LAYER & _layer : __model)
    {
        _layer.differentiate();
    }
}





#endif