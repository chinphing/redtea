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
        class Add;
        class Mul;

        class Param {
            protected :
                bool forwarded;
                MatrixX tensorLoss;
                MatrixX tensorOutput;
            public :
                Param(); 
                bool getForwarded(); 
                void setForwarded(bool f);
                MatrixX& getOutput(); 
                MatrixX& getLoss();
        };

        template<class T>
        class RefVector : public vector<shared_ptr<T>>
        {
            public :
                RefVector();
                RefVector(const RefVector<T>& other);
                RefVector& operator=(const RefVector<T>& other);
        };

        class Tensor {
            protected :
                /*
                * used in forward process to avoid duplicated forking of forward                * method
                */
                shared_ptr<Param> param;
                //input tensors, for back propergation
                RefVector<Tensor> inputs;

            protected :
                Tensor(); 
                Tensor& set(const Tensor& other);
                shared_ptr<Param>& getParam();
                shared_ptr<Param> getParam() const;
                void setParam(const shared_ptr<Param>& param);
                RefVector<Tensor>& getInputs();
                RefVector<Tensor> getInputs() const;
                void setInputs(const RefVector<Tensor>& inputs);
 
            public :
                Tensor(const Tensor& other); 

                virtual shared_ptr<Tensor> copy() const; 

            public :
                /*
                * It will be time efficient if you call this method 
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph. 
                * 
                */
                virtual void reset();
                virtual void forward(); 
                virtual void backward(Optimizer& opti); 

            public :
                MatrixX& getOutput(); 
                MatrixX& getLoss(); 
               
            public :
                Tensor& operator=(const Tensor& other);
                Add operator+(const Tensor& other); 
                Mul operator*(const Tensor& other);
        };

        typedef shared_ptr<Tensor> PTensor;
    };
};


#endif
