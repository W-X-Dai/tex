#include<bits/stdc++.h>
using namespace std;

bool is_prime(int n){
    if(n==1)return false;
    if(n==2)return true;
    for(int i=2;i<n;++i){
        if(n%i==0)return false;
    }
    return true;
}

bool GCD(int a, int b){
    if(a>b)swap(a, b);
    for(int i=a;i>=1;--i)
        if(a%i==0 and b%i==0)return i;
}

int main(){
    int n;
    cin >>n;
}