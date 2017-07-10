#ifndef __TENSOR_OPS_H
#define __TENSOR_OPS_H

#include "Tensor.h"
#include "Optimizer.h"
#include <iostream>

using namespace std;

namespace redtea {
    namespace core {
        class Variable : public Tensor {
            public :
                Variable(const MatrixX& mat) : Tensor() {
                    param->getOutput() = mat;
                }
                Variable(int row, int col) : Tensor(){
                    param->getOutput().resize(row, col);
                }

            public :
                type& operator () (int i, int j) {
                    return param->getOutput()(i, j); 
                }

                void backward(Optimizer& opti) {
                    param->getOutput() -= param->getLoss() * opti.getLearningRate();
                } 
        };

        class Constant : public Variable {
            public :
                Constant(const MatrixX& mat) : Variable(mat) {
                    
                }
                Constant(int row, int col) : Variable(row, col) {
                    
                }
            public :
                void backward(Optimizer& opti) { }
        };
        
        class AddTensorOps : public Tensor {
        public :
            AddTensorOps(PTensor a, PTensor b) : Tensor() {
                inputTensors.push_back(a);
                inputTensors.push_back(b);
            }

        public :
            void forward() {
                inputTensors[0]->forward();
                inputTensors[1]->forward();

                MatrixX& a = inputTensors[0]->getOutput();
                MatrixX& b = inputTensors[1]->getOutput();
                assert(a.cols() == b.cols());

                MatrixX& o = param->getOutput();
                if(a.rows() != b.rows()) {
                    o = a + b.replicate(a.rows(), 1);
                } else {
                    o = a + b;
                }
            }

            void backward(Optimizer& opti) {
                MatrixX& a = inputTensors[0]->getOutput();
                MatrixX& b = inputTensors[1]->getOutput();
 
                MatrixX& aLoss = inputTensors[0]->getLoss();
                MatrixX& bLoss = inputTensors[1]->getLoss();

                aLoss = param->getLoss();
                if(a.rows() != b.rows()) {
                    bLoss = MatrixX::Zero(b.rows(), b.cols());
                    for(int i=0;i<a.rows();i++) {
                        bLoss += aLoss.row(i);             
                    }
                } else {
                    bLoss = param->getLoss();
                }
                
                inputTensors[0]->backward(opti);
                inputTensors[1]->backward(opti);
            }
        };

        class MultTensorOps : public Tensor {
        public :
            MultTensorOps(PTensor a, PTensor b) : Tensor(){
                inputTensors.push_back(a);
                inputTensors.push_back(b);
            }

        public :
            void forward() {
                inputTensors[0]->forward();
                inputTensors[1]->forward();

                param->getOutput() = inputTensors[0]->getOutput()
                                         *inputTensors[1]->getOutput();
            }

            void backward(Optimizer& opti) {
                inputTensors[0]->getLoss() = 
                    param->getLoss()*inputTensors[1]->getOutput().transpose();
                inputTensors[1]->getLoss() = 
                    inputTensors[0]->getOutput().transpose() * param->getLoss();
                
                inputTensors[0]->backward(opti);
                inputTensors[1]->backward(opti);
            }
        };
        
    };
};
#endif
