#include<bits/stdc++.h>
using namespace std;

struct st1 {
    string sid, cid, name;
};
struct st2 {
    string cid, name;
};

inline int hash_func(const string &s) {
    int result=0;
    // calculate hash value
    for(char c:s)
        result=(result+(int)c)%500;
    return result;
}

int main() {
    ifstream f1("reserved_ticket.csv"), f2("name.csv");
    string line, s1, s2;// buffers for parsing
    // utilize vector to store multiple records in same hash bucket
    vector<st1> v1[500];// reserved tickets
    vector<st2> v2[500];// name 
    while(getline(f1, line)){
        // parse CSV line
        stringstream ss(line);
        getline(ss, s1, ',');// sid
        getline(ss, s2, ',');// cid
        // insert record into corresponding hash bucket
        v1[hash_func(s2)].push_back({s1, s2, ""});
    }
    while(getline(f2, line)){
        stringstream ss(line);
        getline(ss, s1, ',');// cid
        getline(ss, s2, ',');// name

        int hashed_result=hash_func(s1);
        if(v1[hashed_result].size()){
            for(auto &rec : v1[hashed_result]){
                if(rec.cid==s1){
                    rec.name=s2;
                }
            }
        }
    }

    cout<<"start\n";
    int rows=0;
    for(int i=0;i<500;i++){
        if(v1[i].size()){
            for(const auto &rec : v1[i]) {
                if(rec.name!="")
                    cout<<rec.sid<<","<<rec.name<<"\n", rows++;
            }
        }
    }
    cout<<"total rows: "<<rows<<"\n";
    // for(int i=0;i<10;i++){
    //     for(int j=0;j<10;++j){
    //         if(j!=9)
    //             cout<<i+10*j<<" & "<<v1[i+10*j].size()<<" & ";
    //         else
    //             cout<<i+10*j<<" & "<<v1[i+10*j].size()<<" \\\\";
    //     }
    //     cout<<"\n";
    // }
    // for(int i=0;i<100;i++){
    //     cout<<i<<" bucket size: "<<v1[i].size()<<"\n";
    // }
    return 0;
}


    // ofstream outfile("hash_table.txt");
    // for(int i=0;i<500;i++){
    //     for(const auto &rec : records[i]) {
    //         outfile<<i<<","<<rec.id<<","<<rec.name<<"\n";
    //     }
    // }
    // outfile.close();