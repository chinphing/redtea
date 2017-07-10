#include <iostream>
#include <stdlib.h>

#include "Tensor.h"
#include "TensorOps.h"
#include "Loss.h"
#include "Optimizer.h"


using namespace std;
using namespace Eigen;
using namespace redtea::core;

int main(int argc, char* argv[]) {

    if(argc < 2) return 1;
    int epoch = atoi(argv[1]);
    cout<<"epoch="<<epoch<<endl;

    Matrix<type, 5, 2> sample;
    sample << 1, 1,
              2, 1,
              3, 2,
              5, 3,
              6, 0;

    Matrix<type, 5, 1> target;
    target << 6, 8, 13, 20, 13;

    shared_ptr<Constant> x(new Constant(sample));
    shared_ptr<Constant> y(new Constant(target));

    shared_ptr<Variable> w(new Variable(MatrixX::Random(2, 1)));
    shared_ptr<Variable> b(new Variable(MatrixX::Random(1, 1)));
    
    shared_ptr<Tensor> mul(new MultTensorOps(x, w));
    shared_ptr<Tensor> add(new AddTensorOps(mul, b));

    LeastSquareLoss loss(add, y);

    cout<<"w: "<<w->getOutput()<<endl;
    cout<<"b: "<<b->getOutput()<<endl;

    Optimizer opti(1e-2);
    for(int i=0;i<epoch;i++) {
        loss.forward();
        loss.backward(opti);
        
        cout<<"o: "<<add->getOutput();
        cout<<", w: "<<w->getOutput();
        cout<<", b: "<<b->getOutput();
        cout<<", l: "<<loss.getTotalLoss()<<endl;
    }

    return 0;
}
