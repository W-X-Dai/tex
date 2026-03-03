#include<bits/stdc++.h>
using namespace std;

int main(){
    int a, ly=0;
    cin >>a;
    
    if(a%4==0)ly=1;
    if(a%100==0)ly=0;
    if(a%400==0)ly=1;
    
    if(ly)cout<<"a leap year\n";
    else cout<<"a normal year\n";
    return 0;
}