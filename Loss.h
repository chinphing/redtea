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
                inputTensors[0]->forward();
                inputTensors[1]->forward();
            } 

            void backward(Optimizer& opti) {
                this->computeLoss();
                inputTensors[0]->getLoss() = param->getLoss();
                inputTensors[0]->backward(opti);
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

                    param->getLoss() = predict - target;

                    MatrixX square = param->getLoss().transpose() 
                                            * param->getLoss();
                    loss = square.sum() / 2;
                }
        };
    };

};

#endif
