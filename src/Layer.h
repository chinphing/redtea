#ifndef __LAYER_H
#define __LAYER_H

#include "Tensor.h"
#include "TensorOps.h"
#include "Activation.h"

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
					this->setParam(o.getParam());
                    inputs.push_back(o.copy());
                }
            public :
                DenseLayer(const DenseLayer& other) {
                    set(other);
                }

                typedef DenseLayer Type;
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Tensor> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
        };
		
		//output class of LstmLayer
		class LstmCell : public Tensor {
			public :
                LstmCell() : Tensor() {}
                LstmCell(PTensor in, PTensor c0, PTensor h0, int outputSize, 
					PTensor w_f, PTensor b_f, 
					PTensor w_i, PTensor b_i,
					PTensor w_c, PTensor b_c,
					PTensor w_o, PTensor b_o) : Tensor() {
                    _init(*in, *c0, *h0, outputSize,
						*w_f, *b_f, *w_i, *b_i, *w_c, *b_c, *w_o, *b_o);
                }

                LstmCell(const Tensor& in, const Tensor& c0, 
					const Tensor& h0, int outputSize,
					const Tensor& w_f, const Tensor& b_f, 
					const Tensor& w_i, const Tensor& b_i,
					const Tensor& w_c, const Tensor& b_c,
					const Tensor& w_o, const Tensor& b_o) : Tensor() {
                    _init(in, c0, h0, outputSize,
						w_f, b_f, w_i, b_i, w_c, b_c, w_o, b_o);
                }

				//reference http://www.jianshu.com/p/9dc9f41f0b29
                void _init(const Tensor& x, const Tensor& c0, 
					const Tensor& h0, int outputSize,
					const Tensor& w_f, const Tensor& b_f, 
					const Tensor& w_i, const Tensor& b_i,
					const Tensor& w_c, const Tensor& b_c,
					const Tensor& w_o, const Tensor& b_o) {
                    
					Sigmoid f = Sigmoid(x * w_f + h0 * w_f + b_f);
					Sigmoid i = Sigmoid(x * w_i + h0 * w_i + b_i);
					Sigmoid o = Sigmoid(x * w_o + h0 * w_o + b_o);
					
					Add updateC = x * w_c + h0 * w_c + b_c;
					Add C = MulElt(f, c0) + MulElt(i, Tanh(updateC));
					MulElt output = MulElt(o, C);
					
                    output.setParam(this->getParam());
                    inputs.push_back(output.copy());
                }
            public :
				typedef LstmCell Type;
                LstmCell(const Type& other) {
                    set(other);
                }
           
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
		};
		
		class LstmLayer : public Tensor{
			
			struct LstmParam : public Param {
				int outputSize;
				shared_ptr<Tensor> w_f, b_f;
				shared_ptr<Tensor> w_i, b_i;
				shared_ptr<Tensor> w_c, b_c;
				shared_ptr<Tensor> w_o, b_o;
				RefVector<LstmCell> outputs;
			};
				
			public :
                LstmLayer(){
					param = shared_ptr<LstmParam>(new Param);
				}
                LstmLayer(const RefVector<Tensor> ins, int outputSize) {
					param = shared_ptr<LstmParam>(new Param);
					assert(ins.size() > 0);
					
					int inputRow = ins[0].rows();
					int inputCol = ins[0].cols();
					w_f = Variable::random(inputCol, outputSize);
					b_f = Variable::random(inputRow, outputSize);
					w_i = Variable::random(inputCol, outputSize);
					b_i = Variable::random(inputRow, outputSize);
					w_c = Variable::random(inputCol, outputSize);
					b_c = Variable::random(inputRow, outputSize);
					w_o = Variable::random(inputCol, outputSize);
					b_o = Variable::random(inputRow, outputSize);
					
					Constant h0 = Constant::zeros();
                }
				
			public :
				typedef LstmLayer Type;
                LstmLayer(const Type& other) {
                    set(other);
                }
           
                shared_ptr<Tensor> copy() const{
                    shared_ptr<Type> c(new Type());
                    c->setParam(this->getParam());
                    c->setInputs(this->getInputs());
                    c->setOptimizer(this->getOptimizer());
                    return c;
                }
				
				Type& operator=(const Type& other) {
                    this->set(other);
                    return *this;
                }
		};

    };
};

#endif
