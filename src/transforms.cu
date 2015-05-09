#include "transforms.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <string>
#include <device_matrix.h>
#include <device_arithmetic.h>
#include <device_math.h>
#include "util.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using namespace ext;

typedef device_matrix<float> mat;
/////////////helper functions//////////////////////

template<typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
	T C;

	__host__ __device__
	linear_index_to_row_index(T C) : C(C) {}
	
	__host__ __device__
	T operator()(T i)
	{
			return i/C;
	}
};

void substractMaxPerRow(mat& x);
mat getRowMax(mat& C);
__global__ void substract_max_per_row(float* const A,float* const rmax, unsigned int rows , unsigned int cols);

void substractMaxPerRow(mat& x) {
	mat rmax = getRowMax(x);

	const int N = 32;
	dim3 grid;
	grid.x = (unsigned int) ceil((float) x.getCols() / N );
	grid.y = (unsigned int) ceil((float) x.getRows() / N );
	dim3 threads(N,N);

	substract_max_per_row<<<grid, threads>>>(x.getData(),rmax.getData() , x.getRows(),x.getCols());
	CCE(cudaDeviceSynchronize());
}


__global__ void substract_max_per_row(float* const A, float * const rmax, unsigned int rows,unsigned int cols){
	int x = blockIdx.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols|| y>= rows)
			return;
	A[x * rows +y] -= rmax[y];
}

mat getRowMax(mat& C)
{
	mat rmax(C.getRows(),1);
	mat At = ~C;
	thrust::device_vector<float>row_indices(C.getRows());
	thrust::device_vector<float>row_results(C.getRows());
	thrust::reduce_by_key
	(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C.getCols())),
	 thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C.getCols())) +C.size(),thrust::device_ptr<float>(At.getData()),row_indices.begin(),
	 thrust::device_ptr<float>(rmax.getData()),thrust::equal_to<float>(),thrust::maximum<float>());
	
	return rmax;
}
//////////////////////////////////////////////
///////TRANSFORMS/////////////

Transforms::Transforms(const Transforms& t):_w(t._w),_i(t._i),_pw(t._pw){}

Transforms::Transforms(const mat& w,const mat& b){
	assert(b.getRows()==1 || b.getCols()==1);
	size_t r=b.getRows(),c=b.getCols();
	if(r==1){r=c;c=1;}
	assert(w.getRows()==r);
	float* h_data=new float[w.size()+b.size()];
	float* b_data=new float[b.size()];
	CCE(cudaMemcpy(h_data,w.getData(),w.size() *sizeof(float) ,cudaMemcpyDeviceToHost));
	CCE(cudaMemcpy(b_data,w.getData(),b.size() *sizeof(float) ,cudaMemcpyDeviceToHost));
	for(size_t t=0;t<b.size();++t)
			h_data[w.size()+t]=b_data[t];
	_w.resize(w.getRows(),w.getCols()+1);
	CCE(cudaMemcpy(_w.getData(),h_data,(w.size()+b.size()) * sizeof(float), cudaMemcpyHostToDevice));
	delete [] b_data;
	delete [] h_data;
	_pw.resize(_w.getRows(),_w.getCols(),0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,float range){
	_w.resize(outputdim,inputdim+1);
	rand_init(_w,range); // uniform distribution
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim+1,0);
}

Transforms::Transforms(size_t inputdim,size_t outputdim,myNnGen& ran){
	_w.resize(outputdim,inputdim+1);
	rand_norm(_w,ran);  // default variance = 0.2 , to change varance head to include/util.h
	_w/=sqrt((float)inputdim);
	_pw.resize(outputdim,inputdim+1,0);
}
size_t Transforms::getInputDim()const{
	return _w.getCols();
}
size_t Transforms::getOutputDim()const{
	return _w.getRows();
}

void Transforms::print(ofstream& out){
	float* h_data = new float[_w.size()];
	CCE(cudaMemcpy( h_data, _w.getData(), _w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for(size_t i=0;i<_w.getRows();++i){
    for(size_t j=0;j<_w.getCols()-1;++j){
                out<<" "<<h_data[_w.getRows()*j+i]; 
            }
            out<<endl;
    }
    out<<"<bias> "<<_w.getRows()<<endl;
    for(size_t t=0;t<_w.getRows();++t)
                out<<" "<<h_data[_w.getRows()*(_w.getCols()-1)+t];
	out << endl;
	delete [] h_data;
}
///////////////////////////////
/////////SIGMOID///////////////

Sigmoid::Sigmoid(const Sigmoid& s): Transforms(s){
}
Sigmoid::Sigmoid(const mat& w, const mat& bias): Transforms(w,bias){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Sigmoid::Sigmoid(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Sigmoid::forward(mat& out,const mat& in,bool train){
	mat _inp(in);
	pushOne(_inp);
	out=sigmoid(_w * _inp);
	if(train){
		_i=in;
	}
}
void Sigmoid::backPropagate(mat& out,const mat& delta, float rate,float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==_i.getCols()) );
	mat withoutBias(_w.getRows(),_w.getCols()-1);
	CCE(cudaMemcpy(withoutBias.getData(),_w.getData(),withoutBias.size() * sizeof(float),cudaMemcpyDeviceToDevice));
	out = _i & ((float)1.0-_i) & (~withoutBias * delta);   // this part need tesing
	// update weight
	mat _inp(_i);
	pushOne(_inp);
	_pw= delta * ~_inp + _pw * momentum;
	rate/=(float)_i.getCols();
	_w -= _pw * rate;
}
void Sigmoid::write(ofstream& out){
	out<<"<sigmoid> "<<_w.getRows()<<" "<<_w.getCols()-1<<endl;
	print(out);
}

///////////////////////////////
///////////SOFTMAX/////////////

Softmax::Softmax(const Softmax& s): Transforms(s){
}
Softmax::Softmax(const mat& w, const mat& bias):Transforms(w,bias){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,float range): Transforms(inputdim,outputdim,range){
}
Softmax::Softmax(size_t inputdim,size_t outputdim,myNnGen& ran): Transforms(inputdim,outputdim,ran){
}
void Softmax::forward(mat& out,const mat& in,bool train){
	mat inp=in;
	pushOne(inp);
	mat z=~(_w * inp);
	substractMaxPerRow(z);
	z=~z; // transpose to column vectors
	mat p(z.getRows(), z.getCols());
	
	thrust::device_ptr<float> zPtr(z.getData());
	thrust::device_ptr<float> pPtr(p.getData());
	thrust::transform(zPtr, zPtr + z.size(),pPtr, func::exp<float>());

	mat sumOfProb =  (mat(p.getRows(), p.getRows(),0) += 1) * p;
	out.resize(_w.getRows(),in.getCols());
	thrust::device_ptr<float> outptr(out.getData());
	thrust::device_ptr<float> sPtr(sumOfProb.getData());
	thrust::transform(pPtr,pPtr+p.size(), sPtr,outptr,thrust::divides<float>());

	if(train){
		_i=in;
	}
}

void Softmax::backPropagate(mat& out,const mat& delta,float rate, float momentum){
	assert( (delta.getRows()==_w.getRows()) && (delta.getCols()==_i.getCols()) );
	mat withoutBias(_w.getRows(),_w.getCols()-1);
	CCE(cudaMemcpy(withoutBias.getData(),_w.getData(),withoutBias.size() * sizeof(float),cudaMemcpyDeviceToDevice));
	out = _i & ((float)1.0-_i) & (~withoutBias * delta);   // this part need tesing
	//update weight
	mat inp(_i);
	pushOne(inp);	
	_pw=delta * ~inp + _pw * momentum;
	rate/=(float)_i.getCols();
	_w-= _pw * rate;
}
void Softmax::write(ofstream& out){
	out<<"<softmax> "<<_w.getRows()<<" "<<_w.getCols()-1<<endl;
	print(out);
}
///////////////////////////////
