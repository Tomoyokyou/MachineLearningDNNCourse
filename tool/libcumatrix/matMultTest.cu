#include <iostream>
#include <vector>
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
#include <thrust/transform_reduce.h>

using namespace std;

typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
  T* h_data = new T [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (T) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
  delete [] h_data;
}

void pushOne(mat& in){
	mat tmp(~in);
	thrust::device_vector<float> dvec(tmp.size()+tmp.getRows());
	thrust::device_ptr<float> mat_ptr(tmp.getData());
	thrust::copy(mat_ptr,mat_ptr+tmp.size(),dvec.begin());
	thrust::device_ptr<float> vec_ptr=dvec.data();
	thrust::fill(vec_ptr+tmp.size()+1,vec_ptr+tmp.size()+1+tmp.getRows(),1);
	tmp.resize(tmp.getRows(),tmp.getCols()+1);
	thrust::device_ptr<float> mat_ptr2(tmp.getData());
	thrust::copy(dvec.begin(),dvec.end(),mat_ptr2);
	//CCE(cudaMemcpy(tmp.getData(),dvec.data(),tmp.size()*sizeof(float),cudaMemcpyDeviceToDevice));
	in = ~tmp;
}

template<typename T>
struct linear_index_to_col_index : public thrust::unary_function<T,T>
{
	T C;

	__host__ __device__
	linear_index_to_col_index(T C) : C(C) {}
	
	__host__ __device__
	T operator()(T i)
	{
			return i/C;
	}
};
void substractMaxPerCol(mat& x);
mat getColMax(mat& C);
__global__ void substract_max_per_col(float* const A,float* const rmax, unsigned int rows , unsigned int cols);

void substractMaxPerCol(mat& x) {
	mat rmax = getColMax(x);

	const int N = 32;
	dim3 grid;
	grid.x = (unsigned int) ceil((float) x.getCols() / N );
	grid.y = (unsigned int) ceil((float) x.getRows() / N );
	dim3 threads(N,N);

	substract_max_per_col<<<grid, threads>>>(x.getData(),rmax.getData() , x.getRows(),x.getCols());
	CCE(cudaDeviceSynchronize());
}


__global__ void substract_max_per_col(float* const A, float * const rmax, unsigned int rows,unsigned int cols){
	int x = blockIdx.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols|| y>= rows)
			return;
	A[x * rows +y] -= rmax[x];
}
mat getColMax(mat& C)
{
	mat rmax(C.getCols(),1);
	thrust::device_vector<float>row_indices(C.getCols());
	thrust::device_vector<float>row_results(C.getCols());
	thrust::reduce_by_key
	(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_col_index<int>(C.getRows())),
	 thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_col_index<int>(C.getRows())) +C.size(),thrust::device_ptr<float>(C.getData()),row_indices.begin(),
	 thrust::device_ptr<float>(rmax.getData()),thrust::equal_to<float>(),thrust::maximum<float>());
	
	return rmax;
}
int main(){

mat A(5,8),B(8,5);
randomInit(A);
randomInit(B);

//testing element-wise operation

mat C(8,1), D(8,1);
randomInit(C);
randomInit(D);

printf("C=\n");
C.print();
printf("D=\n");
D.print();

printf("C & D= \n"); ((C) & (D)).print();

cout<<"row max"<<endl;
A.print();
cout<<endl;
mat out=getColMax(A);
cout<<"result"<<endl;
out.print();
substractMaxPerCol(A);
cout<<endl;
A.print();
cout<<endl;
return 0;
}
