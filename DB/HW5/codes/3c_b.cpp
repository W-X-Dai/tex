#include<bits/stdc++.h>
using namespace std;

struct st1 {
    string sid, cid, name;
};
struct st2 {
    string cid, name;
};

int main() {
    ifstream f1("reserved_ticket.csv"), f2("name.csv");
    string line, s1, s2;// buffers for parsing
    // utilize vector to store multiple records in same hash bucket
    vector<st1> v1;// reserved tickets
    vector<st2> v2;// name 
    while(getline(f1, line)){
        stringstream ss(line);
        getline(ss, s1, ',');
        getline(ss, s2, ',');
        v1.push_back({s1, s2, ""});
    }
    while(getline(f2, line)){
        stringstream ss(line);
        getline(ss, s1, ',');
        getline(ss, s2, ',');
        v2.push_back({s1, s2});
    }

    auto start = chrono::high_resolution_clock::now();

    sort(v1.begin(), v1.end(), [](const st1 &a, const st1 &b){
        return a.cid<b.cid;
    });
    sort(v2.begin(), v2.end(), [](const st2 &a, const st2 &b){
        return a.cid<b.cid;
    });

    int pt1=0, pt2=0;
    while(pt1<v1.size() && pt2<v2.size()){
        if(v1[pt1].cid==v2[pt2].cid){
            v1[pt1].name=v2[pt2].name;
            pt1++;// someone may have multiple tickets, thus only move pt1
        }else if(v1[pt1].cid<v2[pt2].cid){
            pt1++;
        }else{
            pt2++;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Join took: " << ms << " ms\n";


    cout<<"start\n";
    int rows=0;
    for(const auto &rec : v1){
        if(rec.name!="")rows++;
    }
    cout<<"total rows: "<<rows<<"\n";

    return 0;
}


    // ofstream outfile("hash_table.txt");
    // for(int i=0;i<500;i++){
    //     for(const auto &rec : records[i]) {
    //         outfile<<i<<","<<rec.id<<","<<rec.name<<"\n";
    //     }
    // }
    // outfile.close();