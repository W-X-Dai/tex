#include<bits/stdc++.h>
using namespace std;

int main(){
    int arr[1000], n;

    cin >>n;
    for(int i=0; i<n; i++){
        cin >>arr[i];
    }

    for(int i=0;i<n;++i){
        cout<<arr[i]*10<<" ";
    }
}