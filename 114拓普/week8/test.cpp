#include<bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 g(rd());

inline string gen_ant(int n){
    if(n>9 || n<1){
        cout<<"Error: n should be between 1 and 9.\n";
        return "";
    }

    int num[]={1, 2, 3, 4, 5, 6, 7, 8, 9};
    shuffle(num, num+n, g);

    string res="";
    for(int i=0; i<n; i++) res+=to_string(num[i]);
    return res;
}

int main(){
    string sa, sb;
    int a[10], b[10], n;

    cin >>n;
    sa=gen_ant(n);
    cout<<sa;
}