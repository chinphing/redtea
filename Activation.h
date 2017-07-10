#ifndef __ACTIVATION_H
#define __ACTIVATION_H

#include "Tensor.h"
#include "Optimizer.h"
#include <iostream>

using namespace std;

namespace redtea {
    namespace core {
        
        class Sigmoid : public Tensor {
        private :
            double alpha;
        protected :
            Sigmoid(PTensor in, double alpha = 1.0) : Tensor() {
                this->alpha = alpha;
                inputTensors.push_back(in);
            }

        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputTensors[0]->getOutput();
                MatrixX& o = this->getOutput();
                o.resize(in.rows(), in.cols());
     
                for(int i=0;i<in.rows();i++) {
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = 1/(1+exp(-alpha*in(i, j)));
                    }
                }
            }

            void backward(Optimizer& opti) {
                MatrixX& o = this->getOutput();
                MatrixX& l = this->getLoss();
                MatrixX& inLoss = inputTensors[0]->getLoss();

                inLoss.resize(o.rows(), o.cols());
                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {
                        inLoss(i, j) = alpha * o(i, j) * (1.0- o(i, j));
                        inLoss(i, j) *= l(i, j);
                    }
                }
                
                Tensor::backward(opti);    
            }

        public :
            static shared_ptr<Sigmoid> create(PTensor in, double alpha=1.0) {
                return shared_ptr<Sigmoid>(new Sigmoid(in, alpha));
            }
        };

        class Softmax : public Tensor {
        private :
            MatrixX expVals;
            MatrixX expSums;
            vector<int> maxIndexes;
        protected :
            Softmax(PTensor in) : Tensor(){
                inputTensors.push_back(in);
            }
        public :
            void forward() {
                Tensor::forward();

                MatrixX& in = inputTensors[0]->getOutput();
                MatrixX& o = this->getOutput();

                o.resize(in.rows(), in.cols());
                expVals.resize(in.rows(), in.cols());
                expSums.resize(in.rows(), 1);
                maxIndexes.clear();

                for(int i=0;i<in.rows();i++) {
                    int maxIndex = 0;
                    double max = in.row(i).maxCoeff(&maxIndex);
                    double sum= 0;
                    //set expVal and expSum for backward
                    for(int j=0;j<in.cols();j++) {
                        expVals(i, j) = exp(in(i, j) - max);
                        sum += expVals(i, j);
                    }
                    expSums(i, 0) = sum;
                    maxIndexes.push_back(maxIndex);
                    
                    //set ouput
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = expVals(i, j) / sum;
                    }
                }
            }

            void backward(Optimizer& opti) {
                MatrixX& o = this->getOutput();
                MatrixX& l = this->getLoss();
                MatrixX& inLoss = inputTensors[0]->getLoss();

                inLoss.resize(o.rows(), o.cols());
                for(int i=0;i<o.rows();i++) {
                    for(int j=0;j<o.cols();j++) {
                        if(j != maxIndexes[i]) {
                            double sum = expSums(i);
                            inLoss(i, j) = 1.0/sum-expVals(i, j)/(sum*sum);
                            inLoss(i, j) *= expVals(i, j)*l(i, j);
                        } else {
                            inLoss(i, j) = 0;
                        }
                    }
                }

                Tensor::backward(opti);
            }
        public :
            static shared_ptr<Softmax> create(PTensor in) {
                return shared_ptr<Softmax>(new Softmax(in));
            }
        };    
    };
};
#endif
