#include<bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 g(rd()); 

//Generate random integer in [1, 5]
uniform_int_distribution<int> rand_int(1, 5);

//Generate random real number in [0.0, 1.0]
uniform_real_distribution<double> rand_real(0.0, 1.0);

int main(){
    //Example of generating random integers
    cout << "Random Integers in [1, 5]:\n";
    for(int i = 0; i < 10; ++i){
        cout << rand_int(g) << " ";
    }
    cout << "\n\n";

    //Example of generating random real numbers
    cout << "Random Real Numbers in [0.0, 1.0]:\n";
    for(int i = 0; i < 10; ++i){
        cout << rand_real(g) << " ";
    }
    cout << "\n";

    return 0;
}