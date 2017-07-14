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

using namespace std;
using namespace Eigen;
using namespace redtea::core;

int main(int argc, char* argv[]) {

    if(argc < 2) return 1;
    int epoch = atoi(argv[1]);
    cout<<"epoch="<<epoch<<endl;

    //prepare test samples
    Matrix<type, 100, 2> sample;
    sample << -0.017612,14.053064,-1.395634,4.662541,-0.752157,6.538620,-1.322371,7.152853,0.423363,11.054677,0.406704,7.067335,0.667394,12.741452,-2.460150,6.866805,0.569411,9.548755,-0.026632,10.427743,0.850433,6.920334,1.347183,13.175500,1.176813,3.167020,-1.781871,9.097953,-0.566606,5.749003,0.931635,1.589505,-0.024205,6.151823,-0.036453,2.690988,-0.196949,0.444165,1.014459,5.754399,1.985298,3.230619,-1.693453,-0.557540,-0.576525,11.778922,-0.346811,-1.678730,-2.124484,2.672471,1.217916,9.597015,-0.733928,9.098687,-3.642001,-1.618087,0.315985,3.523953,1.416614,9.619232,-0.386323,3.989286,0.556921,8.294984,1.224863,11.587360,-1.347803,-2.406051,1.196604,4.951851,0.275221,9.543647,0.470575,9.332488,-1.889567,9.542662,-1.527893,12.150579,-1.185247,11.309318,-0.445678,3.297303,1.042222,6.105155,-0.618787,10.320986,1.152083,0.548467,0.828534,2.676045,-1.237728,10.549033,-0.683565,-2.166125,0.229456,5.921938,-0.959885,11.555336,0.492911,10.993324,0.184992,8.721488,-0.355715,10.325976,-0.397822,8.058397,0.824839,13.730343,1.507278,5.027866,0.099671,6.835839,-0.344008,10.717485,1.785928,7.718645,-0.918801,11.560217,-0.364009,4.747300,-0.841722,4.119083,0.490426,1.960539,-0.007194,9.075792,0.356107,12.447863,0.342578,12.281162,-0.810823,-1.466018,2.530777,6.476801,1.296683,11.607559,0.475487,12.040035,-0.783277,11.009725,0.074798,11.023650,-1.337472,0.468339,-0.102781,13.763651,-0.147324,2.874846,0.518389,9.887035,1.015399,7.571882,-1.658086,-0.027255,1.319944,2.171228,2.056216,5.019981,-0.851633,4.375691,-1.510047,6.061992,-1.076637,-3.181888,1.821096,10.283990,3.010150,8.401766,-1.099458,1.688274,-0.834872,-1.733869,-0.846637,3.849075,1.400102,12.628781,1.752842,5.468166,0.078557,0.059736,0.089392,-0.715300,1.825662,12.693808,0.197445,9.744638,0.126117,0.922311,-0.679797,1.220530,0.677983,2.556666,0.761349,10.693862,-2.168791,0.143632,1.388610,9.341997,0.317029,14.739025;

    Matrix<type, 100, 1> target;
    target << 0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,0,0;

    MatrixX targetSoftmax = Matrix<type, 100, 2>::Zero(100, 2);
    for(int i=0;i<100;i++) {
        if(target(i, 0) > 0.5) targetSoftmax(i, 1) = 1;
        else targetSoftmax(i, 0) = 1;
    }

    /*
    auto x = Constant::create(sample);
    auto y = Constant::create(targetSoftmax);

    //create the network
    auto w = Variable::create(MatrixX::Random(2, 2));
    auto b = Variable::create(MatrixX::Random(1, 2));
   
    auto mul = Mul::create(x, w);
    auto add = Add::create(mul, b);
    auto act = Softmax::create(add);

    //train
    LogisticLoss loss(act, y);
    Optimizer opti(1e-2);
    for(int i=0;i<epoch;i++) {
        loss.reset();
        loss.forward();
        loss.backward(opti);
        
        cout<<", l: "<<loss.getTotalLoss()<<endl;
    }
    
    cout<<"o: "<<act->getOutput()<<endl;
    */

    cout<<"direct mode."<<endl;
    Constant x(sample);
    Constant y(targetSoftmax);

    //create the network
    Variable w(MatrixX::Random(2, 2));
    Variable b(MatrixX::Random(1, 2));

    Mul mul(x, w);
    Add add(mul, b);
    Softmax act(add);

    //train
    LogisticLoss loss(act, y);

    //AdamOptimizer opti;
    //AdadeltaOptimizer opti;
    //MomentumOptimizer opti;
    SGDOptimizer opti(1e-3);

    opti.minimize(loss);
    for(int i=0;i<epoch;i++) {
        opti.run();

        cout<<"loss: "<<loss.getOutput().sum()<<endl;
    }

    cout<<"o: "<<act.getOutput()<<endl;

    return 0;
}
