#include "dnn.h"
#include "dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <device_matrix.h>
#include "util.h"

#define MAX_EPOCH 10000000

using namespace std;

typedef device_matrix<float> mat;

float computeErrRate(const vector<size_t>& ans, const vector<size_t>& output);
void computeLabel(vector<size_t>& result,const mat& outputMat);

DNN::DNN():_pData(NULL), _learningRate(0.001),_momentum(0), _method(ALL){}
DNN::DNN(Dataset* pData, float learningRate,float momentum,float variance,Init init, const vector<size_t>& v, Method method):_pData(pData), _learningRate(learningRate),_momentum(momentum), _method(method){
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

void DNN::train(size_t batchSize, size_t maxEpoch = MAX_EPOCH, size_t trainSetNum = 10000, size_t validSetNum = 10000, float alpha = 0.98){
	mat trainSet;
	vector<size_t> trainLabel;
	mat validSet;
	vector<size_t> validLabel;
	size_t EinRise = 0;
	float Ein = 1;
	float pastEin = Ein;
	float minEin = Ein;
	float Eout = 1;
	float pastEout = Eout;
	float minEout = Eout;
	
	_pData->getTrainSet(trainSetNum, trainSet, trainLabel);
	_pData->getValidSet(validSetNum, validSet, validLabel);
	size_t num = 0;
	for(; num < maxEpoch; num++){
		mat batchData;
		mat batchLabel;
		mat batchOutput;
		_pData->getBatch(batchSize, batchData, batchLabel);
		
		feedForward(batchOutput, batchData, true);

		mat lastDelta(batchOutput - batchLabel );
		backPropagate(lastDelta, _learningRate, _momentum); //momentum

		
		if( num % 2000 == 0 )
			_learningRate *= alpha;

		if( num % 5000 == 1 ){
			vector<size_t> trainResult;
			vector<size_t> validResult;
			predict(trainResult, trainSet);
			predict(validResult, validSet);
			Ein = computeErrRate(trainLabel, trainResult);
			Eout = computeErrRate(validLabel, validResult);
			
			pastEin  = Ein;
			pastEout = Eout;
			if(minEin > Ein){
				minEin = Ein;
			}
			if(minEout > Eout){
				minEout = Eout;
				cout << "bestMdl: Error at: " << minEout << endl;  
				//if(minEout < 0.5){
					ofstream ofs("best.mdl");
					if (ofs.is_open()){
						for(size_t i = 0; i < _transforms.size(); i++){
							(_transforms.at(i))->write(ofs);
						}
					}
					ofs.close();
				//}
			}
			
			cout.precision(5);
			cout << "Validating error: " << Eout*100 << " %, Training error: " << Ein*100 << " %,  iterations:" << num-1 <<"\n";
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
	//delete [] h_data;
}

void DNN::setDataset(Dataset* pData){
	_pData = pData;
}
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
		//cout << "Transform: " << (_transforms.at(i))->getInputDim() << " " << (_transforms.at(i))->getOutputDim() << endl;
		//cout << "Input feature: " << inputMat.getRows() << endl;
		(_transforms.at(i))->forward(outputMat, tempInputMat, train);
		tempInputMat = outputMat;
	}
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
void computeLabel(vector<size_t>& result,const mat& outputMat){
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
