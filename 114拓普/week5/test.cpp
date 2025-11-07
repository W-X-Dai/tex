//Given n numbers, find the maximum sum of any contiguous subarray (elements in the array may be negative).
#include<bits/stdc++.h>
using namespace std;

int main(){
    int arr[10005], n;
    cin >>n;
    for(int i=0;i<n;++i)cin >>arr[i];

    int ma=-1e9, cur=0;
    for(int i=0;i<n;++i){
        for(int j=i;j<n;++j){
            cur+=arr[j];
            ma=max(ma, cur);
        }
        cur=0;
    }

    cout<<ma<<'\n';
}
/*
7
-2 1 -3 4 1 -1 5
*/