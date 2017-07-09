#ifndef __TENSOR_OPS_H
#define __TENSOR_OPS_H

#include "Tensor.h"
#include "Optimizer.h"

namespace redtea {
    namespace Core {
        class Variable : public Tensor {
            public :
                Variable(const MatrixX& mat) {
                    param->getOutput() = mat;
                }
                Variable(int row, int col) {
                    param->getOutput().resize(row, col);
                }

            public :
                type& operator () (int i, int j) {
                    return param->getOutput()(i, j); 
                }

                void backward(Optimizer& opti) {
                    param->getOutput() -= param->getLoss() * learningRate();
                } 
        }

        class AddTensorOps : public Tensor {
        public :
            AddTensorOps(PTensor a, PTensor b) {
                inputTensors.push_back(a);
                inputTensors.push_back(b);
                param = shared_ptr<Param*>(new Param());
            }

        public :
            void forward() {
                param->getOutput() = inputTensors[0]->getOutput()
                                         +inputTensors[1]->getOutput();
            }

            void backward(Optimizer& opti) {
                inputTensors[0]->getLoss() = param->getLoss();
                inputTensors[1]->getLoss() = param->getLoss();
                
                inputTensors[0]->backward(opti);
                inputTensors[1]->backward(opti);
            }
        };

        class MultTensorOps : public Tensor {
        public :
            MultTensorOps(PTensor a, PTensor b) {
                inputTensors.push_back(a);
                inputTensors.push_back(b);
                param = shared_ptr<Param*>(new Param());
            }

        public :
            void forward() {
                param->getOutput() = inputTensors[0]->getOutput()
                                         *inputTensors[1]->getOutput();
            }

            void backward(Optimizer& opti) {
                inputTensors[0]->getLoss() = 
                    inputTensors[1]->getOuput().transpose() * param->getLoss();
                inputTensors[1]->getLoss() = 
                    inputTensors[0]->getOuput().transpose() * param->getLoss();
                
                inputTensors[0]->backward(opti);
                inputTensors[1]->backward(opti);
            }
        };
        
    };
};
#endif
