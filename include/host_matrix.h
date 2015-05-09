#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H
#include <iostream>
#include <cassert>
#include <string>
#include <iomanip>

using namespace std;

template<class T>
class host_matrix{
public:
	class Transpose {
	public:
		host_matrix<T> operator + (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m._rows,_m._cols);
			host_geam(_m,rhs,result,(T)1.0,(T)1.0,true,false);
			return result;
		}

		host_matrix<T> operator - (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m._rows,_m._cols);
			host_geam(_m,rhs,result,(T)1.0,(T)-1.0,true,false);
			return result;
		}

		host_matrix<T> operator * (const host_matrix<T>& rhs) {
			host_matrix<T> result(_m._cols,rhs._cols);
			host_gemm(_m,rhs,result,(T)1.0,(T)1.0,true,false);
			return result;
		}
		host_matrix<T> operator * (const Transpose rhs) {
			host_matrix<T> result(_m._cols,rhs._rows);
			host_gemm(_m,rhs,result,(T)1.0,T(1.0),true,true);
			return result;
		}
		Transpose(const host_matrix<T>& m): _m(m){}
		const host_matrix<T> _m;
	};

	public:
	
	host_matrix();
	host_matrix(size_t r, size_t c);
	host_matrix(size_t r, size_t c, T value);
	host_matrix(T* data, size_t r,size_t c);
	host_matrix(const host_matrix<T>& src);
	host_matrix(const Transpose& src);

	~host_matrix();

	host_matrix<T>& operator += (T val);
	host_matrix<T> operator + (T val) const;
	
	host_matrix<T>& operator += (const host_matrix<T>& rhs);
	host_matrix<T> operator + (const host_matrix<T>& rhs) const;

	host_matrix<T>& operator += (const Transpose& rhs);
	host_matrix<T> operator + (const Transpose& rhs) const;
	
	host_matrix<T>& operator -= (T val);
	host_matrix<T>& operator - (T val) const;
	
	host_matrix<T>& operator -= (const host_matrix<T>& rhs);
	host_matrix<T> operator - (const host_matrix<T>& rhs);

	host_matrix<T>& operator -= (const Transpose& rhs);
	host_matrix<T> operator - (const Transpose& rhs) const;

	host_matrix<T>& operator /= (T val);
	host_matrix<T> operator / (T val) const;
	
	host_matrix<T>& operator *= (T val);
	host_matrix<T> operator * (T val) const;
	
	host_matrix<T>& operator *= (const host_matrix<T>& rhs);
	host_matrix<T> operator * (const host_matrix<T>& ths) const;

	host_matrix<T>& operator *= (const Transpose& rhs);
	host_matrix<T> operator * (const Transpose& rhs) const;
	
	host_matrix<T>& operator &= (const host_matrix<T>& rhs);
	host_matrix<T> operator & (const host_matrix<T>& rhs) const;

	host_matrix<T>& operator = (const host_matrix<T>& rhs);

	Transpose operator ~ () const;

	void resize(size_t r,size_t c);
	void resize(size_t r,size_t c,T val);
	void print(int precision=5) const;
	
	void fillwith(T val);
	size_t size() const {return _rows*_cols;}
	size_t getRows() const {return _rows;}
	size_t getCols() const {return _cols;}

	T* getData() const {return _data;}
		
private:
	size_t _rows;
	size_t _cols;
	
	T* _data;
};

template<class T>
host_matrix<T> operator + (T val, const host_matrix<T>& m){
	return m + (T) val;
}

template<class T>
host_matrix<T> operator - (T val, const host_matrix<T>& m){
	return (m - (T) val) * -1.0;
}

template<class T>
host_matrix<T> operator * (T val, const host_matrix<T>& m){
	return m * (T) val;
}

template<class T>
void host_gemm(const host_matrix<T>& A,const host_matrix<T>& B, host_matrix<T>& C, T alpha = 1.0, T beta=0.0, bool transA=false, bool transB=false){
	size_t m = A.getRows();
	size_t n = A.getCols();
	if(transA)
		swap(m,n);
	size_t k = B.getRows();
	size_t l = B.getCols();
	if(transB)
		swap(k,l);
	assert(n==k);
	if(beta!=0){
		assert(C.getRows()==m&&C.getCols()==l);
		if(beta!=1.0)
			C*=beta;
	}
	else
		C.resize(m,l);
	bool m_a=false;
	if(alpha!=1.0){m_a=true;}
	
	T* Ad=A.getData();
	T* Bd=B.getData();
	T* Cd=C.getData();
	T element;
	if(alpha!=0.0){
	for(size_t x=0;x<l;x++){
		for(size_t y=0;y<m;++y){
			element=0.0;
			for(size_t z=0;z<n;++z){
				element+=Ad[m*z+y]*Bd[z*l+x];
			}
			if(m_a)
				Cd[x*m+y]+=alpha*element;
			else
				Cd[x*m+y]+=element;
		}
	}
	}
}
template<class T>
void host_geam(const host_matrix<T>& A,const host_matrix<T>& B,host_matrix<T>& C,T alpha=1.0,T beta= 1.0,bool transA=false, bool transB=false){
	size_t m = A.getRows();
	size_t n = A.getCols();
	if(transA)
		swap(m,n);
	size_t k = B.getRows();
	size_t l = B.getCols();
	if(transB)
		swap(k,l);

	assert(m==k&&n==l);

	C.resize(m,n);

	T* Ad=A.getData();
	T* Bd=B.getData();
	T* Cd=C.getData();
	for(size_t t=0;t<m*n;++t){
		Cd[t]=alpha*Ad[t]+beta*Bd[t];
	}
}


template<class T>
host_matrix<T>::host_matrix(){
	_rows=0;_cols=0;
	_data=new T [1]; //dummy
}
template<class T>
host_matrix<T>::host_matrix(size_t r,size_t c){
	_rows=r;_cols=c;
	_data=new T[r*c];
}
template<class T>
host_matrix<T>::host_matrix(size_t r,size_t c,T value){
	_rows=r;_cols=c;
	size_t s=r*c;
	_data=new T[s];
	for(size_t t=0;t<s;++t)
		_data[t]=value;
}
template<class T>
host_matrix<T>::host_matrix(T* data,size_t r,size_t c){
	_rows=r;_cols=c;
	size_t s=r*c;
	_data=new T[s];
	for(size_t t=0;t<s;++t)
		_data[t]=data[t];
}
template<class T>
host_matrix<T>::host_matrix(const host_matrix<T>& src){
	_rows=src._rows;_cols=src._cols;
	size_t s=_rows*_cols;
	_data=new T[s];
	T* tmp=src.getData();
	for(size_t t=0;t<s;++t)
		_data[t]=tmp[t];
}
template<class T>
host_matrix<T>::host_matrix(const Transpose& src){
	_rows=src._m._rows;_cols=src._m._cols;
	size_t s=_rows*_cols;
	T* tmp=src._m.getData();
	_data=new T[s];
	for(size_t t=0;t<s;++t)
		_data[t]=tmp[t];
}

template<class T>
host_matrix<T>::~host_matrix(){
	delete [] _data;
}
template<class T>
host_matrix<T>& host_matrix<T>::operator += (T val){
	size_t s=_rows*_cols;
	for(size_t t=0;t<s;++t)
		_data[t]+=val;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (T val) const{
	host_matrix<T> temp(*this);
	return (temp += val);
}
template<class T>	
host_matrix<T>& host_matrix<T>::operator += (const host_matrix<T>& rhs){
	assert(_rows==rhs._rows && _cols==rhs._cols);
	size_t s=_rows*_cols;
	T* tmp=rhs.getData();
	for(size_t t=0;t<s;++t)
		_data[t]+=tmp[t];
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(_rows,_cols);
	host_geam(*this,rhs,temp,1.0,1.0,false,false);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator += (const typename host_matrix<T>::Transpose& rhs){
	*this = *this + rhs;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator + (const typename host_matrix<T>::Transpose& rhs) const{
	host_matrix<T> temp(*this);
	return (rhs + temp);
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator -= (T val){
	size_t s=_rows*_cols;
	for(size_t t=0;t<s;++t)
		_data[t]-=val;
	return *this;
}
template<class T>
host_matrix<T>& host_matrix<T>::operator - (T val) const{
	host_matrix<T> temp(*this);
	return (temp-=val);
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator -= (const host_matrix<T>& rhs){
	assert(_rows==rhs._rows && _cols==rhs._cols);
	size_t s=_rows*_cols;
	T* tmp=rhs.getData();
	for(size_t t=0;t<s;++t)
		_data[t]-=tmp[t];
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator - (const host_matrix<T>& rhs){
	host_matrix<T> temp(_rows,_cols);
	host_geam(*this,rhs,temp,(T)1.0,(T)-1.0,false,false);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator -= (const typename host_matrix<T>::Transpose& rhs){
	*this = *this - rhs;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator - (const typename host_matrix<T>::Transpose& rhs) const{
	host_matrix<T> temp(_rows,_cols,0);
	host_geam(*this,rhs._m,temp,(T)1.0,(T)-1.0,false,true);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator /= (T val){
	assert(val!=0.0);
	size_t s=_rows*_cols;
	*this *= (T) 1.0/(T)val;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator / (T val) const{
	host_matrix<T> temp(*this);
	return (temp/=val);
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator *= (T val){
	size_t s=_rows*_cols;
	for(size_t t=0;t<s;++t)
		_data[t]*=val;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (T val) const{
	host_matrix<T> temp(*this);
	return (temp*=val);
}

template<class T>
host_matrix<T>& host_matrix<T>::operator *= (const host_matrix<T>& rhs){
	*this = *this * rhs;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(_rows,_cols,0);
	host_gemm(*this,rhs,temp,(T)1.0,(T)0.0,false,false);
	return temp;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator *= (const Transpose& rhs){
	*this = *this * rhs;
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator * (const typename host_matrix::Transpose& rhs) const{
	host_matrix<T> temp(_rows,rhs._m._rows,0);
	host_gemm(*this,rhs._m,temp,(T)1.0,(T)0.0,false,true);
	return temp;
}
	
template<class T>
host_matrix<T>& host_matrix<T>::operator &= (const host_matrix<T>& rhs){
	assert(_rows==rhs._rows&&_cols==rhs._cols);
	size_t s=_rows*_cols;
	T* tmp=rhs.getData();
	for(size_t t=0;t<s;++t)
		_data[t]*=tmp[t];
	return *this;
}
template<class T>
host_matrix<T> host_matrix<T>::operator & (const host_matrix<T>& rhs) const{
	host_matrix<T> temp(*this);
	return temp&=rhs;
}

template<class T>
host_matrix<T>& host_matrix<T>::operator = (const host_matrix<T>& rhs){
	resize(rhs._rows,rhs._cols);
	size_t s=_rows*_cols;
	T* tmp=rhs.getData();
	for(size_t t=0;t<s;++t)
		_data[t]=tmp[t];
	return *this;
}

template<class T>
typename host_matrix<T>::Transpose host_matrix<T>::operator ~ () const{
	return host_matrix<T>::Transpose(*this);
}

template<class T>
void host_matrix<T>::resize(size_t r,size_t c){
	if(r==_rows && c==_cols)
		return;
	_rows=r;
	_cols=c;
	delete [] _data;
	_data=new T[r*c];
}

template<class T>
void host_matrix<T>::resize(size_t r,size_t c,T val){
	this->resize(r,c);
	fillwith(val);	
}

template<class T>
void host_matrix<T>::print(int precision) const{
	for(size_t t=0;t<_rows;++t){
		cout<<fixed<<setprecision(precision);
		for(size_t k=0;k<_cols;++k){
			cout<<setw(9)<<_data[k*_rows+t];
		}
		cout<<endl;
	}
}
	
template<class T>
void host_matrix<T>::fillwith(T val){
	size_t s=_rows*_cols;
	for(size_t t=0;t<s;++t)
		_data[t]=val;
}


#endif
