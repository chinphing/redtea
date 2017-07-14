#ifndef __LOSS_H
#define __LOSS_H

#include <iostream>
#include "Tensor.h"

using namespace std;

namespace redtea {
    namespace core {

        class LeastSquareLoss : public Tensor {
            public :
                LeastSquareLoss() : Tensor() {}
                LeastSquareLoss(PTensor predict, PTensor target) 
                               : Tensor() {
                    inputs.push_back(predict);
                    inputs.push_back(target);
                }
                LeastSquareLoss(Tensor& predict, Tensor& target)
                               : Tensor() {
                    inputs.push_back(predict.copy());
                    inputs.push_back(target.copy());
                }
            public :
                typedef LeastSquareLoss Type;
                typedef shared_ptr<Type> PType;
                LeastSquareLoss(Type& other) {
                    set(other);
                }

                shared_ptr<Tensor> copy() const{
                    cout<<"copt Leaset"<<endl;
                    PType c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
            public :
                void forward() {
                    Tensor::forward();
                    MatrixX predict = inputs[0]->getOutput();
                    MatrixX target = inputs[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());
                    this->getOutput() = (predict - target).array().square();
                }

                void backward(const MatrixX& deltaLoss) {
                    this->addLoss(deltaLoss);

                    MatrixX predict = inputs[0]->getOutput();
                    MatrixX target = inputs[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());

                    MatrixX deltaLossIn = (predict - target).array()
                                               * deltaLoss.array();

                    inputs[0]->backward(deltaLossIn);
                }
        };

        class LogisticLoss : public Tensor {
            public :
                LogisticLoss() : Tensor() {}
                LogisticLoss(PTensor predict, PTensor target)
                               : Tensor() {
                    inputs.push_back(predict);
                    inputs.push_back(target);
                }
                LogisticLoss(Tensor& predict, Tensor& target)
                               : Tensor()  {
                    inputs.push_back(predict.copy());
                    inputs.push_back(target.copy());
                }
            public :
                typedef LogisticLoss Type;
                typedef shared_ptr<Type> PType;
                LogisticLoss(Type& other) {
                    set(other);
                }

                shared_ptr<Tensor> copy() const{
                    PType c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
            public :
                void forward() {
                    Tensor::forward();
                    
                    MatrixX predict = inputs[0]->getOutput();
                    MatrixX target = inputs[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());

                    MatrixX& o = this->getOutput();
                    o.resize(target.rows(), target.cols());
                    for(int i=0;i<o.rows();i++) {
                        for(int j=0;j<o.cols();j++) {
                            type p = predict(i, j);
                            if(abs(target(i, j)-1) < 1e-6) {
                                o(i, j) = -log(p);
                            } else {
                                o(i, j) = -log(1-p);
                            }
                            
                        }
                    }
                }
                void backward(const MatrixX& deltaLoss) {
                    this->addLoss(deltaLoss);

                    MatrixX predict = inputs[0]->getOutput();
                    MatrixX target = inputs[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());

                    MatrixX deltaLossIn;
                    deltaLossIn.resize(target.rows(), target.cols());
                    for(int i=0;i<deltaLossIn.rows();i++) {
                        for(int j=0;j<deltaLossIn.cols();j++) {
                            type p = predict(i, j);
                            if(abs(target(i, j)-1) < 1e-6) {
                                deltaLossIn(i, j) = -1.0/p;
                            } else {
                                deltaLossIn(i, j) = 1.0/(1-p);
                            }
                            deltaLossIn(i, j) *= deltaLoss(i, j);
                        }
                    }

                    inputs[0]->backward(deltaLossIn);
                }
        };
    };

};

#endif
