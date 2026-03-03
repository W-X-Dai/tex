#include<bits/stdc++.h>
using namespace std;

int main(){
    random_device rd;
    mt19937 g(rd()); 
    uniform_real_distribution<double> rand_real(0.0, 1.0);
    int N=10000000, inside_circle=0;
    for(int i=0;i<N;i++){
        double x=rand_real(g);
        double y=rand_real(g);
        if(x*x+y*y<=1.0) inside_circle++;
    }
    double pi_estimate=4.0*inside_circle/N;
    cout<<"Estimated value of Pi: "<<pi_estimate<<"\n";
    return 0;
}