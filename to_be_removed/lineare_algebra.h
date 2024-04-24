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
        double operator* (vector<T> & w)
        {
            double scalar=0;
            if(this->size() != w.size())throw std::invalid_argument("Dimensionality of provided vector doesn't match.");
            for(int i=0; i < this->size(); i++)
            {
                scalar += this->at(i) * w.at(i);
            }
            return scalar;
        }

        vector<T> operator* (vector<vector<T>> W)
        {
            
        }
    };
} // namespace lineare_algebra




#endif