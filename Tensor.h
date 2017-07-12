#ifndef __TENSOR_H
#define __TENSOR_H

#include <vector>
#include <memory>

#include "def.h"
#include "Optimizer.h"

using namespace Eigen;
using namespace std;

namespace redtea{
    namespace core{
        class Param {
            protected :
                bool forwarded;
                MatrixX tensorLoss;
                MatrixX tensorOutput;
            public :
                Param() {
                    forwarded = false;
                }
                bool getForwarded() {
                    return forwarded;
                }
                void setForwarded(bool f) {
                    forwarded = f;
                }
                 
                virtual MatrixX& getOutput() {
                    return tensorOutput;
                }

                virtual MatrixX& getLoss() {
                    return tensorLoss;
                }
        };

        class Tensor {
            protected :
                /*
                * used in forward process to avoid duplicated forking of forward                * method
                */
                shared_ptr<Param> param;
                //input tensors, for back propergation
                vector<shared_ptr<Tensor>> inputTensors;

            protected :
                Tensor() {
                    param = shared_ptr<Param>(new Param); 
                }
                shared_ptr<Param>& getParam(){
                    return param;
                }
                vector<shared_ptr<Tensor>>& getInputs(){
                    return inputTensors;
                }
            public :
                Tensor(Tensor& other) {
                    set(other);
                }

                typedef Tensor Type;
                typedef shared_ptr<Type> PType;
                virtual shared_ptr<Tensor> copy() {
                    return PType(new Type(*this));
                }

            protected :
                Tensor& set(Tensor& other) {
                    this->param = other.getParam();
                    this->inputTensors.assign(
                          other.getInputs().begin(), other.getInputs().end());
                    return *this;
                }
            public :
                /*
                * It will be time efficient if you call this method 
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph. 
                * 
                */
                virtual void reset() {
                    param->setForwarded(false);
                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->reset();
                    }
                }
                virtual void forward() {
                    if(param->getForwarded()) return;

                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->forward();
                    }
                    param->setForwarded(true);
                }

                virtual void backward(Optimizer& opti) {
                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->backward(opti);
                    }
                }

            public :
                MatrixX& getOutput() {
                    return param->getOutput();
                }

                MatrixX& getLoss() {
                    return param->getLoss();
                }
                
        };

        typedef shared_ptr<Tensor> PTensor;
    };
};


#endif
