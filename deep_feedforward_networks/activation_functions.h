#ifndef ACTIVATION_FUNCTIONS_INCLUDE_GUARD
#define ACTIVATION_FUNCTIONS_INCLUDE_GUARD

#include<functional>
#include<string>

namespace activation_functions
{
    int heaviside_step(int value)
    /**
     * Implements the heaviside_step function used in the very beginnings of deep learning.
     * Simulates the behaviour of a biological neuron.
     * @cite https://en.wikipedia.org/wiki/Heaviside_step_function
    */
    {
        return value>=0;
    }
}


#endif