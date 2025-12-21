#include<iostream>
#include<random>
#include<memory>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

random_device rd;
mt19937 gen(rd());
normal_distribution<> dist(0.0, 0.1);

class ANN{
    struct Layer{
        virtual VectorXd forward(const VectorXd &x)=0;
        virtual VectorXd backward(const VectorXd &x){return x;};
        virtual void update(double lr){};
        virtual ~Layer(){}
    };

    struct linear:Layer {
        int in_dim, out_dim;

        MatrixXd W;// weight
        MatrixXd delta;// for update
        VectorXd b;// bias
        VectorXd a_lazy;// for backward

        linear(int in_d, int out_d)
            :in_dim(in_d), out_dim(out_d), W(MatrixXd::Random(out_d, in_d)), b(VectorXd::Zero(out_d))
        {}

        VectorXd forward(const VectorXd &x) override{
            a_lazy=x;
            return W*x+b;
        }

        VectorXd backward(const VectorXd &x) override{
            delta=x;
            return W.transpose()*x;
        }

        void update(double lr) override{
            W=W-lr*(delta*a_lazy.transpose());
            b=b-lr*delta;
        }
    };

    struct sigmoid:Layer{
        VectorXd a_lazy;

        VectorXd forward(const VectorXd &x) override{
            a_lazy=x.unaryExpr([](double a){
                return 1.0/(1.0+exp(-a));
            });
            return a_lazy;
        }

        VectorXd backward(const VectorXd &x) override{
            return a_lazy.array()*(1-a_lazy.array())*x.array();
        }
    };

    struct ReLU:Layer{
        VectorXd a_lazy;

        VectorXd forward(const VectorXd &x) override{
            a_lazy=x.unaryExpr([](double a){
                return max(a, 0.0);
            });
            return a_lazy;
        }

        VectorXd backward(const VectorXd &x) override{
            VectorXd result(x.size());
            for(int i=0;i<x.size();++i){
                result[i]=(a_lazy[i]>0 ? x[i] : 0.0);
            }
            return result;
        }        
    };

    double lr;
    vector<unique_ptr<Layer>> layers;


public:
    ANN(double lrate): lr(lrate){}

    void add_linear(int in_dim, int out_dim){
        layers.emplace_back(make_unique<linear>(in_dim, out_dim));
    }

    void add_sigmoid(){
        layers.emplace_back(make_unique<sigmoid>());
    }

    void add_relu(){
        layers.emplace_back(make_unique<ReLU>());
    }

    void update(){
        for(auto &L:layers){
            L->update(lr);
        }
    }

    VectorXd forward(const VectorXd &input){
        VectorXd out=input;
        for(auto &L: layers){
            out=L->forward(out);
        }
        return out;
    }

    void backward(const double &y_true, const double &y_pred){
        VectorXd dA(1);
        dA << y_pred-y_true;
        for(int i=layers.size()-1;i>=0;--i){
            dA=layers[i]->backward(dA);
        }
    }

    double BCE(const double &y_true, const double &y_pred){
        return -(y_true*log(y_pred)+(1-y_true)*log(1-y_pred));
    }

};

int main(){
    ANN ann(0.1);
    ann.add_linear(2,3);
    ann.add_sigmoid();
    ann.add_linear(3,1);
    ann.add_sigmoid();

    vector<VectorXd> X = { (VectorXd(2)<<0,0).finished(),
                           (VectorXd(2)<<0,1).finished(),
                           (VectorXd(2)<<1,0).finished(),
                           (VectorXd(2)<<1,1).finished() };
    vector<double> Y = {0,1,1,0};

    int n_epoch=10000;
    for(int e=0;e<n_epoch;++e){
        double epo_loss=0;

        for(int i=0;i<4;++i){
            auto pred = ann.forward(X[i]);
            double loss = ann.BCE(Y[i], pred(0));
            epo_loss+=loss;
            ann.backward(Y[i], pred(0));
            ann.update();
        }

        if(e%1000==0 or e==n_epoch-1){
            cout<<"[epoch "<<e<<" ] Loss: "<<epo_loss/4<<'\n';
        }
    }

    for(auto &x:X){
        cout << ann.forward(x) << endl;
    }
}