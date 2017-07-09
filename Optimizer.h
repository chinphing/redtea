#ifndef __TENSOR_H
#define __TENSOR_H

#include <list>
#include <def.h>

using namespace std;

namespace redtea{
    namespace core{

        class Optimizer {
            protected :
                double learningRate;

            public :
                Optimizer(double learningRate) {
                    this->learningRate = learningRate;
                }

                double getLearningRate() {
                    return learningRate;
                }
        };
    };
};


#endif
