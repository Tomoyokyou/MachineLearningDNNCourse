#include "mynngen.h"
#include <device_matrix.h>
#include <device_arithmetic.h>
#include <device_math.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>

typedef device_matrix<float> mat;

myNnGen gn(0,0.2);

void rand_init(mat& w,float range){
	float* h_data = new float[w.size()];
	for(size_t t=0;t<w.getRows()*(w.getCols()-1);++t)
		h_data[t]=2*range*rand()/(float)RAND_MAX - range;
	for(size_t t=0;t<w.getRows();++t)
		h_data[t+w.getRows()*(w.getCols()-1)]=0;
	CCE(cudaMemcpy(w.getData(),h_data,w.size()* sizeof(float) , cudaMemcpyHostToDevice));
	delete [] h_data;
}
void rand_norm(mat& w,myNnGen& ran){
	float* h_data = new float[w.size()];
	for(size_t t=0;t<w.getRows()*(w.getCols()-1);++t)
		h_data[t]=ran();
	for(size_t t=0;t<w.getRows();++t)
		h_data[t+w.getRows()*(w.getCols()-1)]=0;
	CCE(cudaMemcpy(w.getData(),h_data,w.size()* sizeof(float) , cudaMemcpyHostToDevice));
	delete [] h_data;
}

void pushOne(mat& in){
	mat tmp(~in);
	thrust::device_vector<float> dvec(tmp.size()+(tmp.getRows()+1),1);
	thrust::device_ptr<float> mat_ptr(tmp.getData());
	thrust::copy(mat_ptr,mat_ptr+tmp.size(),dvec.begin());
	thrust::device_ptr<float> vec_ptr=dvec.data();
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	thrust::device_ptr<float> mat_ptr2(tmp.getData());
	thrust::copy(dvec.begin(),dvec.end(),mat_ptr2);
	in = ~tmp;
}

void getBias(mat& out,const mat& w){
	float* h_data=new float[w.getRows()];
	CCE(cudaMemcpy(h_data,w.getData()+(w.getRows())*(w.getCols()-1),sizeof(float)*w.getRows(),cudaMemcpyDeviceToHost));
	out.resize(w.getRows(),1);
	CCE(cudaMemcpy(out.getData(),h_data,sizeof(float)*w.getRows(),cudaMemcpyHostToDevice));
	delete [] h_data;
}

void replaceBias(mat& w,const mat& bias){
	assert(bias.getCols()==1);
	assert(w.getRows()==bias.size());
	CCE(cudaMemcpy(w.getData()+w.getRows()*(w.getCols()-1),bias.getData(),sizeof(float)*w.getRows(),cudaMemcpyDeviceToDevice));
}

void parseDim(string str,vector<size_t>& dim){
	size_t begin=str.find_first_not_of(' '),end;
	string hold;
	while(begin!=string::npos){
		end=str.find_first_of('-',begin);
		if(end==string::npos)
			hold=str.substr(begin);
		else
			hold=str.substr(begin,end-begin);
		if(!hold.empty())
			dim.push_back(atoi(hold.c_str()));
		begin=str.find_first_not_of('-',end);
	}
}

