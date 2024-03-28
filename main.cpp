#include"deep_feedforward_networks/perceptron.h"
#include<iostream>

int main()
{
    PERCEPTRON p(2);
    p.train({{0,0},{0,1},{1,0},{1,1}},{0,1,1,0},100,0.01);
    std::cout<<p.predict({0,0})<<'\n';
    std::cout<<p.predict({1,0})<<'\n';
    std::cout<<p.predict({0,1})<<'\n';
    std::cout<<p.predict({1,1})<<'\n';
}