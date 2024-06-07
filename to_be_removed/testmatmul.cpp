#include<iostream>
#include "matmul.h"
#include <time.h>
#include<vector>

void mmm(std::vector<double> & data1, std::vector<double> & data2, std::vector<int> & shape1, std::vector<int> & shape2)
{
    std::vector<double> result;

    for (int i = 0; i < shape1[0]; i++) // replace this 
    {
        for (int j = 0; j < shape2[1]; j++)
        {
            double sum = 0;
            for (int k = 0; k < shape1[1]; k++)
            {
                sum += data1[i * shape1[1] + k] * data2[k * shape2[1] + j];
            }
            result.push_back(sum);
        }
    }

    data1.swap(result);
    shape1 = {shape1[0], shape2[1]};
}

int main()
{
    srand(time(0));
    int N = 1e4;
    double slowest = 0;
    int a, b, c;
    for (int k = 0; k < 1; k++)
    { 
        int i = rand()%100+9900;
        int w1 = rand()%10+90;
        int w2 = rand()%10+90;
        std::vector<double> data1(i*w1);
        std::vector<double> data2(w1*w2);
        for (int j = 0; j < i*w1; j++)
        {
            data1[j] = rand();
        }
        for (int j = 0; j < w1*w2; j++)
        {
            data2[j] = rand();
        }
        std::vector<int> shape1 = {i, w1};
        std::vector<int> shape2 = {w1, w2};
        MATMUL m;
        clock_t tStart = clock();
        m.matmul(data1, data2, shape1, shape2);
        //printf("%d, %d, %d: ", i, w1, w2);
        //printf("Time taken: %.3fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        if (((double)(clock() - tStart)/CLOCKS_PER_SEC) > slowest) 
        {
            slowest = ((double)(clock() - tStart)/CLOCKS_PER_SEC);
            a = i;
            b = w1;
            c = w2;
        }
    }
    printf("%.3f \n%d, %d, %d", slowest, a, b, c);
}