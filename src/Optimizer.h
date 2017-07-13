#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

#include <iostream>
#include <list>
#include "def.h"

using namespace std;
using namespace Eigen;

namespace redtea{
    namespace core{
        class Optimizer {
            public :
                virtual shared_ptr<Optimizer> copy() const = 0;
                virtual void update(MatrixX& param, MatrixX& loss) = 0;
        };

        class SGDOptimizer : public Optimizer{
            protected :
                double learningRate;

            public :
                SGDOptimizer() {
                    learningRate = 1e-3;
                }
                SGDOptimizer(double learningRate) {
                    this->learningRate = learningRate;
                }

                double setLearningRate(double l) {
                    learningRate = l;
                }
                double getLearningRate() const{
                    return learningRate;
                }

                typedef SGDOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setLearningRate(this->getLearningRate());
                    return c;
                }

                void update(MatrixX& param, MatrixX& loss) {
                    param -= loss * learningRate;
                }
        };

        class MomentumOptimizer : public Optimizer {
            protected :
                double rho;
                double learningRate;
                MatrixX delta;

            public :
                MomentumOptimizer() {
                    rho = 0.95;
                    learningRate = 1e-3;
                }
                MomentumOptimizer(double r, double l) {
                    rho = r;
                    learningRate = l;
                }
 
                double getRho() const{
                    return rho;
                }

                double getLearningRate() const{
                    return learningRate;
                }

                typedef MomentumOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(
                        new Type(
                            this->getRho(), this->getLearningRate() ));
                    return c;
                }
 
                void update(MatrixX& param, MatrixX& loss) {
                    if(delta.rows() <= 0) 
                        delta = MatrixX::Zero(param.rows(), param.cols());
                    
                    delta = rho * delta -learningRate * loss;
                    param += delta;
                }
        };

        class AdadeltaOptimizer : public Optimizer {
            protected :
                double rho;
                double epsilon;
                MatrixX egs;
                MatrixX exs;

            public :
                AdadeltaOptimizer() {
                    rho = 0.95;
                    epsilon = 1e-6;
                }
                AdadeltaOptimizer(double r, double e) {
                    rho = r;
                    epsilon = e;
                }

                double getRho() const{
                    return rho;
                }

                double getEpsilon()  const {
                    return epsilon;
                }

                typedef AdadeltaOptimizer Type;
                shared_ptr<Optimizer> copy() const{
                    shared_ptr<Type> c(
                        new Type(
                            this->getRho(), this->getEpsilon() ));
                    return c;
                }

                void update(MatrixX& param, MatrixX& loss) {
                    if(egs.rows() <= 0) {
                        egs = MatrixX::Zero(param.rows(), param.cols());
                    }
                    if(exs.rows() <= 0) {
                        exs = MatrixX::Zero(param.rows(), param.cols());
                    }

                    MatrixX temp = loss.array().square();
                    egs = rho * egs + (1.0-rho)*temp;

                    double learningRate = 0.0;
                    MatrixX delta(param.rows(), param.cols());
                    for(int i=0;i<param.rows();i++) {
                        for(int j=0;j<param.cols();j++) {
                            learningRate = sqrt(exs(i, j)+epsilon) 
                                               / sqrt(egs(i, j)+epsilon);
                            //cout<<"learningRate: "<<learningRate<<endl;
                            delta(i, j) = learningRate * loss(i, j);
                        }
                    }

                    temp = delta.array().square();
                    exs = rho * exs + (1.0-rho) * temp;
                    
                    param -= delta;
                }
        };
    };
};


#endif
