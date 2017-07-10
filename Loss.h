#ifndef __LOSS_H
#define __LOSS_H

#include "Tensor.h"

namespace redtea {
    namespace core {
        class Loss : public Tensor {
        protected :
            double loss;
        public : 
            Loss(PTensor predict, PTensor target) {
                loss = 0;
                inputTensors.push_back(predict);
                inputTensors.push_back(target);
            }

            double getTotalLoss() {
                return loss;
            }

            virtual void computeLoss() {
                
            }
            
            void forward() {
                Tensor::forward();
            } 

            void backward(Optimizer& opti) {
                this->computeLoss();
                inputTensors[0]->getLoss() = this->getLoss();
                Tensor::backward(opti);
            }
        };

        class LeastSquareLoss : public Loss {
            public :
                LeastSquareLoss(PTensor predict, PTensor target) 
                               : Loss(predict, target) {}

                void computeLoss() {
                    MatrixX predict = inputTensors[0]->getOutput();
                    MatrixX target = inputTensors[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());

                    this->getLoss() = predict - target;

                    MatrixX square = this->getLoss().transpose() 
                                            * this->getLoss();
                    loss = square.sum() / 2;
                }
        };

        class LogisticLoss : public Loss {
            public :
                LogisticLoss(PTensor predict, PTensor target)
                               : Loss(predict, target) {}

                void computeLoss() {
                    loss = 0.0;
                    MatrixX predict = inputTensors[0]->getOutput();
                    MatrixX target = inputTensors[1]->getOutput();
                    assert(predict.cols() == target.cols()
                           && predict.rows() == target.rows());
                    
                    MatrixX& l = this->getLoss();
                    l.resize(target.rows(), target.cols());
                    for(int i=0;i<l.rows();i++) {
                        for(int j=0;j<l.cols();j++) {
                            type p = predict(i, j);
                            if(abs(target(i, j)-1) < 1e-6) {
                                loss += -log(p);
                                l(i, j) = -1.0/p;
                            } else {
                                loss += -log(1-p);
                                l(i, j) = 1.0/(1-p);
                            }
                        }
                    }
                    
                }
        };
    };

};

#endif
