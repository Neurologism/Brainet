#ifndef LAYER_INCLUDE_GUARD
#define LAYER_INCLUDE_GUARD

#include <vector>

/**
 * @brief Base class determining the properties each layer should have.
*/
class LAYER
{
protected:
    std::vector<double> __input; 
    
    LAYER(){};
    ~LAYER();
    virtual std::vector<double> operation(std::vector<double>&) =0; // forward pass over layer
    virtual void differentiate() =0;   // calculate part of the chain rule
    virtual std::vector<double> backpropagation(std::vector<double>&) =0; // backward pass over the layer 
};

#endif // LAYER_INCLUDE_GUARD