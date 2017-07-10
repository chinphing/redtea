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
                MatrixX& o = param->getOutput();
                o.resize(in.rows(), in.cols());
     
                for(int i=0;i<in.rows();i++) {
                    for(int j=0;j<in.cols();j++) {
                        o(i, j) = 1/(1+exp(-alpha*in(i, j)));
                    }
                }
            }

            void backward(Optimizer& opti) {
                MatrixX& o = param->getOutput();
                MatrixX& l = param->getLoss();
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
        
    };
};
#endif
