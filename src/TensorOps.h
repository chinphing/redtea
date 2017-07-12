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
                Variable(const Variable& other) {
                    set(other);
                }

                typedef Variable Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    return c;
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
                Constant(const Constant& other) {
                    set(other);
                }
 
                typedef Constant Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    return c;
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
            Add() : Tensor() {}
            Add(PTensor a, PTensor b) : Tensor() {
                inputs.push_back(a);
                inputs.push_back(b);
            }

            Add(const Tensor& a, const Tensor& b) {
                inputs.push_back(a.copy());
                inputs.push_back(b.copy());
            }
        public :
                Add(const Add& other) {
                    set(other);
                }

                typedef Add Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    return c;
                }

                Add& operator=(const Add& other) {
                    this->set(other);
                    return *this;
                }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& a = inputs[0]->getOutput();
                MatrixX& b = inputs[1]->getOutput();
                assert(a.cols() == b.cols());

                MatrixX& o = this->getOutput();
                if(a.rows() != b.rows()) {
                    o = a + b.replicate(a.rows(), 1);
                } else {
                    o = a + b;
                }
            }

            void backward(Optimizer& opti) {
                MatrixX& a = inputs[0]->getOutput();
                MatrixX& b = inputs[1]->getOutput();
 
                MatrixX& aLoss = inputs[0]->getLoss();
                MatrixX& bLoss = inputs[1]->getLoss();

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
            Mul() : Tensor() {}
            Mul(PTensor a, PTensor b) : Tensor(){
                inputs.push_back(a);
                inputs.push_back(b);
            }
            Mul(const Tensor& a, const Tensor& b) : Tensor(){
                inputs.push_back(a.copy());
                inputs.push_back(b.copy());
            }
        public :
                Mul(const Mul& other) {
                    set(other);
                }
                
                typedef Mul Type;
                shared_ptr<Tensor> copy() const {
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    return c;
                }
                
                Mul& operator=(const Mul& other) {
                    set(other);
                    return *this;
                }
        public :
            void forward() {
                Tensor::forward();
                
                this->getOutput() = inputs[0]->getOutput()
                                         *inputs[1]->getOutput();
            }

            void backward(Optimizer& opti) {
                inputs[0]->getLoss() = 
                    this->getLoss()*inputs[1]->getOutput().transpose();
                inputs[1]->getLoss() = 
                    inputs[0]->getOutput().transpose() * this->getLoss();
                
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
