#ifndef __SHAPE_OPS_H
#define __SHAPE_OPS_H

class SubTensor : public Tensor {
            public :
                SubTensor() : Tensor() { }
                SubTensor(const ) : Tensor() {
                    setRows(mat.rows());
                    setCols(mat.cols());
                }
                SubTensor(int row, int col) : Tensor(){
                    this->getOutput().resize(row, col);
                    setRows(row);
                    setCols(col);
                }
            public :
                Split(const SubTensor& other) {
                    set(other);
                }

                typedef SubTensor Type;
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
            public :
                void backward(const MatrixX& deltaLoss) {
                    this->addLoss(deltaLoss);
                }
                void update() {
                    optimizer->update(this->getOutput(), this->getLoss());
                } 
            public :
                static shared_ptr<Variable> create(const MatrixX& mat) {
                    return shared_ptr<Variable>(new Variable(mat));
                }
                static shared_ptr<Variable> create(int row, int col) {
                    return shared_ptr<Variable>(new Variable(row, col));
                }
				static Variable random(int row, int col) {
					return Variable(MatrixX::Random(row,col));
				}
        };


#endif
