#include<bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 g(rd());

inline string gen_ans(int n){
    if(n>9 || n<2){
        cout<<"Error: n should be between 2 and 9.\n";
        return "";
    }

    string num="123456789";
    shuffle(num.begin(), num.end(), g);

    string res="";
    for(int i=0; i<n; i++) res+=num[i];
    return res;
}

inline bool check_ipnut(string &input, int n){
    if(input.length()!=n){
        cout<<"Error: input length should be "<<n<<".\n";
        return false;
    }
    for(int i=0; i<n; i++){
        char c=input[i];
        if(c<'1' || c>'9'){
            cout<<"Error: input should only contain digits from 1 to 9.\n";
            return false;
        }
    }
    for(int i=0; i<n; i++){
        for(int j=i+1; j<n; j++){
            if(input[i]==input[j]){
                cout<<"Error: input should not contain duplicate digits.\n";
                return false;
            }
        }
    }
    return true;
}

inline int A(string &a, string &b, int n){
    int A_cnt=0;
    for(int i=0; i<n; i++){
        if(a[i]==b[i]) A_cnt++;
    }
    return A_cnt;
}

inline int B(string &a, string &b, int n){
    int B_cnt=0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i!=j && a[i]==b[j]) B_cnt++;
        }
    }
    return B_cnt;
}

int main(){
    string ans, input;
    int a[10], b[10], n;

    cin >>n;
    ans=gen_ans(n);
    
    while(input!=ans){
        cin >>input;
        if(!check_ipnut(input, n)) continue;
        cout<<A(ans, input, n)<<"A"<<B(ans, input, n)<<"B\n";
    }
    cout<<"Congratulations! You found the answer: "<<ans<<"\n";
    return 0;
}