/*
* test case for LeastSquareLoss
*/

#include <iostream>
#include <stdlib.h>

#include "Tensor.h"
#include "TensorOps.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Activation.h"

#include "data.h"

using namespace std;
using namespace Eigen;
using namespace redtea::core;

int main(int argc, char* argv[]) {

    if(argc < 2) return 1;
    int epoch = atoi(argv[1]);
    cout<<"epoch="<<epoch<<endl;

    MatrixX sample;
    MatrixX target;
    loadCsv("./test/lstm-data.csv", sample, target);

    cout<<"direct mode"<<endl;
    Constant x(sample);
    Constant y(target);

    Variable w(MatrixX::Random(sample.cols(), 1));
    Variable b(MatrixX::Random(1, 1));

    Mul mul(x, w);
    Add add(mul, b);
    Sigmoid act(add);
   
    LogisticLoss loss(act, y);

    //AdamOptimizer opti;
    SGDOptimizer opti(1e-3);
    //AdadeltaOptimizer opti;
    //MomentumOptimizer opti(0.8, 1e-3);

    opti.minimize(loss);

    for(int i=0;i<epoch;i++) {
        opti.run();
        if(i % 100 == 0) {
            cout<<"loss: "<<loss.getOutput().mean()<<endl;
        }
    }
    cout<<"o: "<<act.getOutput()<<endl; 
    cout<<"w: "<<w.getOutput()<<endl;
    cout<<"b: "<<b.getOutput()<<endl;
    
    return 0;
}
