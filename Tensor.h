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
                MatrixX loss;
                MatrixX output;
            public :
                virtual MatrixX& getOutput() {
                    return output;
                }

                virtual MatrixX& getLoss() {
                    return loss;
                }
        };

        class Tensor {
            protected :
                //porinter to a inner tensor, every tensor class should implement a inner tensor
                shared_ptr<Param> param;
                
                //input tensors, for back propergation
                vector<shared_ptr<Tensor>> inputTensors;

            public :
                Tensor() {
                    param = shared_ptr<Param>(new Param());
                }

                virtual void forward() {
    
                }

                virtual void backward(Optimizer& opti) {

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
