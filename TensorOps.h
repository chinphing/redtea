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
                Variable() : Tensor() { }
                Variable(const MatrixX& mat) : Tensor() {
                    this->getOutput() = mat;
                }
                Variable(int row, int col) : Tensor(){
                    this->getOutput().resize(row, col);
                }
            public :
                Variable(Variable& other) {
                    set(other);
                }

                typedef Variable Type;
                typedef shared_ptr<Type> PType;
                shared_ptr<Tensor> copy() {
                    return PType(new Type(*this));
                }

            public :
                type& operator () (int i, int j) {
                    return this->getOutput()(i, j); 
                }

                void backward(Optimizer& opti) {
                    this->getOutput() -= this->getLoss() * opti.getLearningRate();
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
                Constant() : Variable() {}
                Constant(const MatrixX& mat) : Variable(mat) {
                    
                }
                Constant(int row, int col) : Variable(row, col) {
                    
                }
            public :
                Constant(Constant& other) {
                    set(other);
                }

                typedef Constant Type;
                typedef shared_ptr<Type> PType;
                shared_ptr<Tensor> copy() {
                    return PType(new Type(*this));
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

            Add(Tensor& a, Tensor& b) {
                inputTensors.push_back(a.copy());
                inputTensors.push_back(b.copy());
            }
        public :
                Add(Add& other) {
                    set(other);
                }

                typedef Add Type;
                typedef shared_ptr<Type> PType;
                shared_ptr<Tensor> copy() {
                    return PType(new Type(*this));
                }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& a = inputTensors[0]->getOutput();
                MatrixX& b = inputTensors[1]->getOutput();
                assert(a.cols() == b.cols());

                MatrixX& o = this->getOutput();
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

                aLoss = this->getLoss();
                if(a.rows() != b.rows()) {
                    bLoss = MatrixX::Zero(b.rows(), b.cols());
                    for(int i=0;i<a.rows();i++) {
                        bLoss += aLoss.row(i);             
                    }
                } else {
                    bLoss = this->getLoss();
                }

                Tensor::backward(opti);    
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
            Mul(Tensor& a, Tensor& b) : Tensor(){
                inputTensors.push_back(a.copy());
                inputTensors.push_back(b.copy());
            }
        public :
                typedef Mul Type;
                typedef shared_ptr<Type> PType;
                Mul(Type& other) {
                    set(other);
                }

                shared_ptr<Tensor> copy() {
                    return PType(new Type(*this));
                }
        public :
            void forward() {
                Tensor::forward();
                
                this->getOutput() = inputTensors[0]->getOutput()
                                         *inputTensors[1]->getOutput();
            }

            void backward(Optimizer& opti) {
                inputTensors[0]->getLoss() = 
                    this->getLoss()*inputTensors[1]->getOutput().transpose();
                inputTensors[1]->getLoss() = 
                    inputTensors[0]->getOutput().transpose() * this->getLoss();
                
                Tensor::backward(opti);    
            }

        public :
            static shared_ptr<Mul> create(PTensor a, PTensor b) {
                return shared_ptr<Mul>(new Mul(a, b));
            }
        };
        
    };
};
#endif
