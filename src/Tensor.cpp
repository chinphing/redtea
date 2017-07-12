#include "Tensor.h"

#include <vector>
#include <memory>

#include "def.h"
#include "Optimizer.h"
#include "TensorOps.h"

using namespace Eigen;
using namespace std;

namespace redtea{
    namespace core{
                Param::Param() {
                    forwarded = false;
                }
                bool Param::getForwarded() {
                    return forwarded;
                }
                void Param::setForwarded(bool f) {
                    forwarded = f;
                }

                MatrixX& Param::getOutput() {
                    return tensorOutput;
                }

                MatrixX& Param::getLoss() {
                    return tensorLoss;
                }


                template<class T>
                RefVector<T>::RefVector() : vector<shared_ptr<T>>() {} 
             
                template<class T>
                RefVector<T>::RefVector(const RefVector<T>& other) 
                    : vector<shared_ptr<T>>() {
                    this->assign(other.begin(), other.end());
                }

                template<class T>
                RefVector<T>& RefVector<T>::operator=(
                                  const RefVector<T>& other) {
                    this->assign(other.begin(), other.end());
                    return *this;
                }

                Tensor::Tensor() {
                    param = shared_ptr<Param>(new Param);
                }
                shared_ptr<Param>& Tensor::getParam() {
                    return param;
                }
                shared_ptr<Param> Tensor::getParam() const {
                    return param;
                }
                void Tensor::setParam(const shared_ptr<Param>& param) {
                    this->param = param;
                }
                RefVector<Tensor>& Tensor::getInputs(){
                    return inputs;
                }
                RefVector<Tensor> Tensor::getInputs() const {
                    return inputs;
                }
                void Tensor::setInputs(const RefVector<Tensor>& inputs) {
                    this->inputs = inputs;
                }


                Tensor::Tensor(const Tensor& other) {
                    set(other);
                }
                
                typedef Tensor Type;
                shared_ptr<Tensor> Tensor::copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    return c;
                }

                Tensor& Tensor::set(const Tensor& other) {
                    this->param = other.getParam();
                    this->inputs = other.getInputs();
                    return *this;
                }

                /*
                * It will be time efficient if you call this method
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph.
                *
                */
                void Tensor::reset() {
                    param->setForwarded(false);
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->reset();
                    }
                }
                void Tensor::forward() {
                    cout<<"tensor forward."<<endl;
                    if(param->getForwarded()) return;

                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->forward();
                    }
                    param->setForwarded(true);
                }

                void Tensor::backward(Optimizer& opti) {
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->backward(opti);
                    }
                }

                MatrixX& Tensor::getOutput() {
                    return param->getOutput();
                }

                MatrixX& Tensor::getLoss() {
                    return param->getLoss();
                }

                Tensor& Tensor::operator=(const Tensor& other) {
                    set(other);
                    return *this;
                }
                
                Add Tensor::operator+(const Tensor& other) {
                    Add add(*this, other);
                    return add;
                }
                Mul Tensor::operator*(const Tensor& other) {
                    Mul mul(*this, other);
                    return mul;
                }
    };
};
