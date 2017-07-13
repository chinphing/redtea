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
                    updated = false;
                }
                bool Param::getForwarded() {
                    return forwarded;
                }
                void Param::setForwarded(bool f) {
                    forwarded = f;
                }
                bool Param::getUpdated() {
                    return updated;
                }
                bool Param::setUpdated(bool u) {
                    updated = u;
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
                shared_ptr<Optimizer> Tensor::getOptimizer() const {
                    return optimizer;
                }
                void Tensor::setOptimizer(const shared_ptr<Optimizer>& opti) {
                    this->optimizer = opti;
                }
                void Tensor::setOptimizer(const Optimizer& opti) {
                    if(optimizer) return;

                    optimizer = opti.copy();
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->setOptimizer(opti);
                    }
                }

                Tensor::Tensor(const Tensor& other) {
                    set(other);
                }
                
                typedef Tensor Type;
                shared_ptr<Tensor> Tensor::copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }

                Tensor& Tensor::set(const Tensor& other) {
                    this->param = other.getParam();
                    this->inputs = other.getInputs();
                    this->optimizer = other.getOptimizer();
                    return *this;
                }

                

                /*
                * It will be time efficient if you call this method
                * when there are more than one collections for                 *                * a Tensor Object in the Tensor graph.
                *
                */
                void Tensor::reset() {
                    param->setForwarded(false);
                    param->setUpdated(false);

                    MatrixX& loss = param->getLoss();
                    if(loss.rows() > 0) loss = MatrixX::Zero(
                                            loss.rows(), loss.cols());
 
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->reset();
                    }
                }
                void Tensor::forward() {
                    if(param->getForwarded()) return;

                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->forward();
                    }
                    param->setForwarded(true);
                }

                void Tensor::backward(const MatrixX& deltaLoss) {
                }

                void Tensor::update() {
                    if(param->getUpdated()) return;
                    for(int i=0;i<inputs.size();i++) {
                        inputs[i]->update();
                    }
                    param->setUpdated(true);
                }

                void Tensor::setOutput(const MatrixX& output) {
                    param->getOutput() = output;
                }

                MatrixX& Tensor::getOutput() {
                    return param->getOutput();
                }

                void Tensor::addLoss(const MatrixX& deltaLoss) {
                    MatrixX& loss = param->getLoss();
                    if(loss.rows() <= 0) loss = MatrixX::Zero(
                                          deltaLoss.rows(), deltaLoss.cols()); 
                    loss += deltaLoss;
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
