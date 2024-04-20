#include "lineare_algebra.h"
#include "deep_feedforward_networks/perceptron.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    la::vector<int> v, w;
    vector<int> v1({8, 2}), w1({3, 4});
    v.swap(v1);
    w.swap(w1);
    cout << v * w;
}