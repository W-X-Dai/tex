#include<bits/stdc++.h>
using namespace std;

int main(){
    for(int i=1;i<=9;i++){          // outer loop for rows
        for(int j=1;j<=9;j++){      // inner loop for columns
            cout<<i<<"x"<<j<<"="<<i*j<<'\t';       // print the product
        }
        cout<<'\n';                  // new line after each row
    }
}