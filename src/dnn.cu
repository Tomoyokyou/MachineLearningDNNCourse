#include "dnn.h"
#include "util.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <ctime>
#include <device_matrix.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

#define MAX_EPOCH 10000000

using namespace std;

typedef device_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);

DNN::DNN():_learningRate(0.001),_momentum(0), _method(ALL){}
DNN::DNN(float learningRate, float momentum, float variance,Init init, const vector<size_t>& v, Method method):_learningRate(learningRate), _momentum(momentum), _method(method){
	int numOfLayers = v.size();
	switch(init){
	case NORMAL:
		gn.reset(0,variance);
		for(int i = 0; i < numOfLayers-1; i++){
			Transforms* pTransform;
			if( i < numOfLayers-2 )
				pTransform = new Sigmoid(v.at(i), v.at(i+1), gn);
			else
				pTransform = new Softmax(v.at(i), v.at(i+1), gn);
			_transforms.push_back(pTransform);
		}
		break;
	case UNIFORM:
	case RBM:
	default:
		for(int i = 0; i < numOfLayers-1; i++){
			Transforms* pTransform;
			if( i < numOfLayers-2 )
				pTransform = new Sigmoid(v.at(i), v.at(i+1), variance);
			else
				pTransform = new Softmax(v.at(i), v.at(i+1), variance);
			_transforms.push_back(pTransform);
		}
		break;
	}
}
DNN::~DNN(){
	while(!_transforms.empty()){
		delete _transforms.back();
		_transforms.pop_back();
	}
}

void DNN::train(Dataset& labeledData, size_t batchSize, size_t maxEpoch = MAX_EPOCH, float trainRatio = 0.8, float alpha = 0.98){
	//clock_t rt1 = clock();

	if(labeledData.isLabeled() == false){
		cerr << "It is impossible to train unLabeled data.\n";
		return;
	}

	Dataset trainData;
	Dataset validData;

	labeledData.dataSegment(trainRatio, trainData, validData);

	//mat trainSet;
	//vector<size_t> trainLabel;
	mat validSet = validData.getData();
	vector<size_t> validLabel = validData.getLabel();

	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	
	//_pData->getTrainSet(trainSetNum, trainSet, trainLabel);
	//_pData->getValidSet(validSetNum, validSet, validLabel);

	//clock_t rt2 = clock();
	//cout << "Get train/validate set:" << (rt2-rt1)/CLOCKS_PER_SEC << endl;
	
	size_t oneEpoch = trainData.getDataNum();
	size_t num = 0;
	for(; num < maxEpoch; num++){
		//clock_t rt3 = clock();
		mat batchData;
		mat batchLabel;

		trainData.getBatch(batchSize, batchData, batchLabel, true);
		mat batchOutput;
		//_pData->getBatch(batchSize, batchData, batchLabel);
		
		//clock_t rt4 = clock();
		feedForward(batchOutput, batchData, true);
		// DEBUG PART
		/*
		vector<size_t> debug;
		float ERR=1.0;
		predict(debug, trainSet);
		ERR= computeErrRate(trainLabel,debug);
		if(ERR==1.0){
			cout<<"iter"<<num<<" encounter 100\% error"<<endl;
				cerr<<"ERROR: program overflow..."<<endl;
				ofstream core("dnn.dump");
				for(size_t t=0;t<_transforms.size();++t)
					_transforms.at(t)->dump(core);
					core<<"\n last output:\n";
					batchOutput.print(core);
					core<<"\n";
					core<<"first gradient "<<endl;
					mat tempOut(batchOutput-batchLabel);
					tempOut.print(core);
					core<<endl;
			exit(1);
		}
		*/
		//

		//clock_t rt5 = clock();
		mat lastDelta(batchOutput - batchLabel);
		backPropagate(lastDelta, _learningRate, _momentum); //momentum

		//clock_t rt6 = clock();	
		
		if( num % 2000 == 0 )
			_learningRate *= alpha;

		if( num % 5000 == 1 ){

			//clock_t rt7 = clock();
			//vector<size_t> trainResult;
			vector<size_t> validResult;
			//predict(trainResult, trainSet);
			predict(validResult, validSet);

			//clock_t rt8 = clock();
			//Ein = computeErrRate(trainLabel, trainResult);
			Eout = computeErrRate(validLabel, validResult);
			
			//clock_t rt9 = clock();

			/*Print debug message here*/
			//double duration = (rt9-rt3);
			//cout << "Per iteration: " << duration/CLOCKS_PER_SEC << " sec\n";
			//cout << "Get Batch time: " << (rt4-rt3)/duration << endl;
			//cout << "Feedforward: " << (rt5-rt4)/duration << endl;
			//cout << "Backpropagation: " << (rt6-rt5)/duration << endl;
			//cout << "Predict train/valid err: " << (rt8-rt7)/duration << endl;
			//cout << "Compute train/valid err: " << (rt9-rt8)/duration << endl;


			//pastEin  = Ein;
			pastEout = Eout;
			//if(minEin > Ein){
			//	minEin = Ein;
			//}
			if(minEout > Eout){
				minEout = Eout;
				cout << "bestMdl: Error at: " << minEout << endl;  
				if(minEout < 0.5){
					ofstream ofs("best.mdl");
					if (ofs.is_open()){
						for(size_t i = 0; i < _transforms.size(); i++){
							(_transforms.at(i))->write(ofs);
						}
					}
					ofs.close();
				}
			}
			
			cout.precision(4);
			//cout << "Validating error: " << Eout*100 << " %, Training error: " << Ein*100 << " %,  iterations:" << num-1 <<"\n";
			cout << "Validating error: " << Eout*100 << " %,  iterations:" << num-1 <<"\n";
		}
	}
	cout << "Finished training for " << num << " iterations.\n";
	cout << "bestMdl: Error at: " << minEout << endl;  
}

void DNN::predict(vector<size_t>& result, const mat& inputMat){
	mat outputMat(1, 1);
	feedForward(outputMat, inputMat, false);
	computeLabel(result, outputMat);
	/*  Transpose matrix print.
	for(size_t i = 0; i < outputMat.getRows(); i++){
		for(size_t j = 0; j < outputMat.getCols(); j++){
			cout << h_data[j*outputMat.getRows() + i] << " ";
		}
		cout << endl;
	}
	
	cout << endl;
	*/
}

//void DNN::setDataset(Dataset* pData){
//	_pData = pData;
//}
void DNN::setLearningRate(float learningRate){
	_learningRate = learningRate;
}
void DNN::setMomentum(float momentum){
	_momentum = momentum;
}

size_t DNN::getInputDimension(){
	return _transforms.front()->getInputDim();
}

size_t DNN::getOutputDimension(){
	return _transforms.back()->getOutputDim();
}

size_t DNN::getNumLayers(){
	return _transforms.size()+1;
}

void DNN::save(const string& fn){
	ofstream ofs(fn);
	if (ofs.is_open()){
		for(size_t i = 0; i < _transforms.size(); i++){
			(_transforms.at(i))->write(ofs);
		}
	}
	ofs.close();
}

bool DNN::load(const string& fn){
	ifstream ifs(fn);
	char buf[50000];
	if(!ifs){return false;}
	else{
		while(ifs.getline(buf, sizeof(buf)) != 0 ){
			string tempStr(buf);
			size_t found = tempStr.find_first_of(">");
			if(found !=std::string::npos ){
				size_t typeBegin = tempStr.find_first_of("<") + 1;
				string type = tempStr.substr(typeBegin, 7);
				stringstream ss(tempStr.substr(found+1));
				string rows, cols;
				size_t rowNum, colNum;
				ss >> rows >> cols;
				rowNum = stoi(rows);
				colNum = stoi(cols);
				size_t totalEle = rowNum * colNum;
				float* h_data = new float[totalEle];
				float* h_data_bias = new float[rowNum];
				for(size_t i = 0; i < rowNum; i++){
					if(ifs.getline(buf, sizeof(buf)) == 0){
						cerr << "Wrong file format!\n";
					}
					tempStr.assign(buf);
					stringstream ss1(tempStr);	
					for(size_t j = 0; j < colNum; j++){
						ss1 >> h_data[ j*rowNum + i ];
					}
				}
				ifs.getline(buf, sizeof(buf));
				ifs.getline(buf, sizeof(buf));
				tempStr.assign(buf);
				stringstream ss2(tempStr);
				float temp;
				for(size_t i = 0; i < rowNum; i++){
					ss2 >> h_data_bias[i];
				}
				mat weightMat(rowNum, colNum);
				mat biasMat(rowNum, 1);		
				cudaMemcpy(weightMat.getData(), h_data, totalEle * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(biasMat.getData(), h_data_bias, rowNum * sizeof(float), cudaMemcpyHostToDevice);
				
				Transforms* pTransform;
				if(type == "sigmoid")
					pTransform = new Sigmoid(weightMat, biasMat);
				else if(type == "softmax")
					pTransform = new Softmax(weightMat, biasMat);
				else{
					cerr << "Undefined activation function! \" " << type << " \"\n";
					exit(1);
				}
				_transforms.push_back(pTransform);
				delete [] h_data;
				delete [] h_data_bias;
			}
		}
	}
	ifs.close();
	return true;
}

void DNN::feedForward(mat& outputMat, const mat& inputMat, bool train){
	mat tempInputMat = inputMat;
	for(size_t i = 0; i < _transforms.size(); i++){
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
}

void DNN::getHiddenForward(mat& outputMat, const mat& inputMat){
	_transforms.at(0)->forward(outputMat, inputMat, false);
}

//The delta of last layer = _sigoutdiff & grad(errorFunc())
void DNN::backPropagate(const mat& deltaMat, float learningRate, float momentum){
	mat tempMat = deltaMat;
	mat errorMat;
	for(int i = _transforms.size()-1; i >= 0; i--){
		(_transforms.at(i))->backPropagate(errorMat, tempMat, learningRate, momentum);
		tempMat = errorMat;
	}
}

//Helper Functions
size_t countDifference(const mat& m1, const mat& m2) {
	assert(m1.size() == m2.size());
	
	size_t L = m1.size();
  	thrust::device_ptr<float> ptr1(m1.getData());
 	thrust::device_ptr<float> ptr2(m2.getData());

  	size_t nDiff = thrust::inner_product(ptr1, ptr1 + L, ptr2, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  	return nDiff;
}

void computeLabel(vector<size_t>& result,const mat& outputMat){

	//int data[6] = {1, 0, 2, 2, 1, 3};
	//int result = thrust::reduce(thrust::host, data, data + 6, -1, thrust::maximum<int>()); // result == 3
	//thrust::device_vector<float>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());

	//unsigned int position = iter - d_vec.begin();
	//float max_val = *iter;
	
	size_t inputDim = outputMat.getRows();
	size_t featureNum = outputMat.getCols();
	thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(outputMat.getData());
	thrust::host_vector<float> h_vec(d_ptr, d_ptr + inputDim*featureNum);
	for(size_t j = 0; j < outputMat.getCols(); j++){
		thrust::host_vector<float>::iterator iter = thrust::max_element(h_vec.begin() + j*inputDim, h_vec.begin() + (j+1)*inputDim);
		unsigned int position = iter - h_vec.begin() - j*inputDim;
		result.push_back(position);
	}

	/*
	float* h_data = new float [outputMat.size()];
	cudaMemcpy(h_data ,outputMat.getData(), outputMat.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for(size_t j = 0; j < outputMat.getCols(); j++){
		float tempMax = h_data[j*outputMat.getRows()];
		size_t idx = 0;		
		for(size_t i = 0; i < outputMat.getRows(); i++){
			if(tempMax < h_data[j*outputMat.getRows() + i]){
				tempMax = h_data[j*outputMat.getRows() + i];
				idx = i;
			}
		}
		result.push_back(idx);
	}
	delete [] h_data;
	*/
}

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output){
	assert(ans.size() == output.size());
	size_t accCount = 0;
	for(size_t i = 0; i < ans.size(); i++){
		if(ans.at(i) == output.at(i)){
			accCount++;
		}
	}
	return 1.0-(float)accCount/(float)ans.size();
}
