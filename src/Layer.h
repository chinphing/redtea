#ifndef __LAYER_H
#define __LAYER_H

#include "Tensor.h"
#include "TensorOps.h"

namespace redtea {
    namespace core {

class DenseLayer : public Tensor {
    public :
        DenseLayer() : Tensor() {}
        DenseLayer(PTensor in, int outputSize) : Tensor() {
            _init(*in, outputSize);
        }

        DenseLayer(const Tensor& in, int outputSize) : Tensor() {
            _init(in, outputSize);
        }

        void _init(const Tensor& x, int outputSize) {
            Variable w(MatrixX::Random(x.cols(), outputSize));
            Variable b(MatrixX::Random(1, outputSize));
            Add o = x * w + b;

            //directly output to the DenseLayer class
            o.setParam(this->getParam());

            inputs.push_back(o.copy());
        }
    public :
        DenseLayer(const DenseLayer& other) {
            set(other);
        }

        typedef DenseLayer Type;
        shared_ptr<Tensor> copy() const{
            shared_ptr<Type> c(new Type());
            c->setParam(this->getParam());
            c->setInputs(this->getInputs());
            c->setOptimizer(this->getOptimizer());
            return c;
        }
};

};
};

#endif
