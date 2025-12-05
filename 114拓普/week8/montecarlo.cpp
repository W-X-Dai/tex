#include<bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 g(rd());

uniform_real_distribution<double> dist(0, 1);

int main(){
    int n=1000, r=1000;
    for(int i=0;i<n;++i){
        cout<<dist(g)<<endl;
    }

}