#include "tensor.h"

using namespace std;

// implementin general tests
int main()
{
    vector<TENSOR<double>> v;
    for(int i=0;i<100;i++)
    {
        TENSOR<double> a({2,2});
        v.push_back(a);
    }
    for(int i=0;i<100;i++)
    {
        v[i].~TENSOR();
        
    } 
    v.clear();   
    return 0;
}