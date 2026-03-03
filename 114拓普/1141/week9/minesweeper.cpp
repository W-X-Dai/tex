#include<bits/stdc++.h>
using namespace std;

int n, game_map[12][12], record[12][12];//game map: n*n
int n_mines, n_blank;
int dir[8][2]={{1, 0}, {0, 1}, {-1, 0}, {0, -1},
                {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

random_device rd;
mt19937 g(rd()); 
uniform_int_distribution<int> rand_int(1, 2);

inline bool initialize(){
    cout<<"Please choose the size of the map(6~10)\n>>>";
    cin >>n;
    if(n>5 and n<=10){
        return 0;
    }else{
        cout<<"Input should be in thea range of [6, 10]!\n";
        return 1;
    }
}

inline void gen_map(){
    for(int i=1;i<=n;++i)
        for(int j=1;j<=n;++j)
            if(rand_int(g)==1)game_map[i][j]=9, ++n_mines;
    n_blank=n*n-n_mines;
    for(int i=1;i<=n;++i)
        for(int j=1;j<=n;++j)
            if(game_map[i][j]==9)
                for(int d=0;d<8;++d)
                    if(game_map[i+dir[d][0]][j+dir[d][1]]!=9)
                        ++game_map[i+dir[d][0]][j+dir[d][1]];
}

inline void show_ans(){
    cout<<"There are "<<n_mines<<" mines\n";
    cout<<"There are "<<n_blank<<" blanks\n";
    for(int i=1;i<=n;++i){
        for(int j=1;j<=n;++j){
            if(game_map[i][j]==9)cout<<"*\t";
            else cout<<game_map[i][j]<<"\t";
        }
        cout<<"\n";
    }
}

inline void show_map(){
    for(int i=1;i<=n;++i){
        for(int j=1;j<=n;++j){
            if(record[i][j])cout<<game_map[i][j];
            else cout<<"["<<i<<" "<<j<<"]";
            cout<<"\t";
        }
        cout<<'\n';
    }
}

int main(){
    while(initialize());
    gen_map();
    //show_ans();
    show_map();
    while(1){
        int x, y;
        cout<<"Where do you want to step?\n>>>";
        cin >>x>>y;
        record[x][y]=1;
        show_map();
        if(game_map[x][y]==9){
            cout<<"Here is a landmine here!\n";
            cout<<"YOU LOSE !\n";
            break;
        }
    }
    return 0;
}