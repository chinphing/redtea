#ifndef __LOSS_H
#define __LOSS_H

#include "Tensor.h"

namespace redtea {
    namespace core {
        class Loss : public Tensor {
        public : 
            Loss(PTensor predict, PTensor target) {
                inputTensors.push_back(predict);
                inputTensors.push_back(target);
            }


            virtual void computeLoss() {
                
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
                    assert(predict.cols() == 1 && target.cols() == 1 
                           && predict.rows() == target.rows());

                    param->getLoss() = target - predict;
                }
        };
    };

};

#endif
