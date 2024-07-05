#include "tensor.h"

using namespace std;

// implementin general tests
int main()
{
    vector<TENSOR<double>> v;
    for(int i=0;i<100;i++)
    {
        TENSOR<double> a({2,2});
        a.set({0,0},1);
        a.set({1,0},2);
        a.set({0,1},3);
        a.set({1,1},4);
        a.at({0,0});
        a.data();
        a.dimensionality();
        a.shape();
        a.size();
        a.transpose();
        v.push_back(a);
    }
    for(int i=0;i<100;i++)
    {
        v[i].at({0,0});
        v[i].data();
        v[i].dimensionality();
        v[i].shape();
        v[i].size();
        v[i].transpose();
    }
    return 0;
}