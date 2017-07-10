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
                MatrixX tensorLoss;
                MatrixX tensorOutput;

                //input tensors, for back propergation
                vector<shared_ptr<Tensor>> inputTensors;

            protected :
                Tensor() {
                }

            public :
                virtual void forward() {
                    for(int i=0;i<inputTensors.size();i++) {
                        inputTensors[i]->forward();
                    }
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
