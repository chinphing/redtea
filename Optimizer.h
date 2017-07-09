#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

#include <list>
#include "def.h"

using namespace std;
using namespace Eigen;

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
