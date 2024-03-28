#ifndef ACTIVATION_FUNCTIONS_INCLUDE_GUARD
#define ACTIVATION_FUNCTIONS_INCLUDE_GUARD


namespace activation_functions
{
    /**
     * Implements the heaviside_step function used in the very beginnings of deep learning.
     * Simulates the behaviour of a biological neuron.
     * @cite https://en.wikipedia.org/wiki/Heaviside_step_function
    */
    double heaviside_step(double value)
    {
        return value>=0;
    }

  
    /**
     * @attention ReLU function
     * g(z)=max(0,z)
    */
    double ReLU(double value)
    {
        return std::max(value,0.0);
    }


} // namespace activation_functions


#endif