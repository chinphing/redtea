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
            public :
                static shared_ptr<Variable> create(const MatrixX& mat) {
                    return shared_ptr<Variable>(new Variable(mat));
                }
                static shared_ptr<Variable> create(int row, int col) {
                    return shared_ptr<Variable>(new Variable(row, col));
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
            public :
                static shared_ptr<Variable> create(const MatrixX& mat) {
                    return shared_ptr<Constant>(new Constant(mat));
                }
                static shared_ptr<Variable> create(int row, int col) {
                    return shared_ptr<Constant>(new Constant(row, col));
                }
        };
        
        class Add : public Tensor {
        public :
            Add(PTensor a, PTensor b) : Tensor() {
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

        public :
            static shared_ptr<Add> create(PTensor a, PTensor b) {
                return shared_ptr<Add>(new Add(a, b));
            }
        };

        class Mul : public Tensor {
        public :
            Mul(PTensor a, PTensor b) : Tensor(){
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

        public :
            static shared_ptr<Mul> create(PTensor a, PTensor b) {
                return shared_ptr<Mul>(new Mul(a, b));
            }
        };
        
    };
};
#endif
