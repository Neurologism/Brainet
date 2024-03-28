#ifndef LINEARE_ALGEBRA_INCLUDE_GUARD
#define LINEARE_ALGEBRA_INCLUDE_GUARD

#include<vector>
#include<stdexcept>

namespace la
{
    template <class T>
    class vector : public std::vector<T>
    {
    public:
        vector(T[] v)
        {
            
        }


        double operator* (vector<T> & w)
        {
            double scalar=0;
            if(this->size() != w.size())throw std::invalid_argument("Dimensionality of provided vector doesn't match.");
            for(int i=0; i < this->size(); i++)
            {
                scalar += this->operator[i] * w[i];
            }
            return scalar;
        }
    };
} // namespace lineare_algebra




#endif