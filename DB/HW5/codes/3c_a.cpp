#include<bits/stdc++.h>
using namespace std;
using namespace std::chrono;

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

    vector<st1> v1;// reserved tickets
    vector<st2> v2;// name
    vector<st1> hash1[500];// reserved tickets
    vector<st2> hash2[500];// name 
    while(getline(f1, line)){
        // parse CSV line
        stringstream ss(line);
        getline(ss, s1, ',');// sid
        getline(ss, s2, ',');// cid
        // insert record into corresponding hash bucket
        v1.push_back({s1, s2, ""});
    }
    while(getline(f2, line)){
        stringstream ss(line);
        getline(ss, s1, ',');// cid
        getline(ss, s2, ',');// name
        v2.push_back({s1, s2});
    }

    auto start = chrono::high_resolution_clock::now();

    for(auto &rec:v1){
        hash1[hash_func(rec.cid)].push_back(rec);
    }
    for(auto &rec:v2){
        if(hash1[hash_func(rec.cid)].size()){
            for(auto &r : hash1[hash_func(rec.cid)]){
                if(r.cid==rec.cid){
                    r.name=rec.name;
                }
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();
    cout << "Join took: " << ms << " ms\n";

    
    cout<<"start\n";
    int rows=0;
    for(int i=0;i<500;i++){
        if(hash1[i].size()){
            for(const auto &rec : hash1[i]) {
                if(rec.name!="")rows++;
            }
        }
    }
    cout<<"total rows: "<<rows<<"\n";
    return 0;
}