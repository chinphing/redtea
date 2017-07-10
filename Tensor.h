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

        class Tensor {
            protected :
                /*
                * used in forward process to avoid duplicated forking of forward                * method
                */
                bool forwarded;

                MatrixX tensorLoss;
                MatrixX tensorOutput;

                //input tensors, for back propergation
                vector<shared_ptr<Tensor>> inputTensors;

            protected :
                Tensor() {
                    forwarded = false; 
                }

            public :
                /*
                * It will be time efficient if you call this method 
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph. 
                * 
                */
                virtual void reset() {
                    forwarded = false;
                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->reset();
                    }
                }
                virtual void forward() {
                    if(forwarded) return;

                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->forward();
                    }
                    forwarded = true;
                }

                virtual void backward(Optimizer& opti) {
                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->backward(opti);
                    }
                }

            public :
                MatrixX& getOutput() {
                    return tensorOutput;
                }

                MatrixX& getLoss() {
                    return tensorLoss;
                }
                 
        };

        typedef shared_ptr<Tensor> PTensor;
    };
};


#endif
