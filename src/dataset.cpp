#include "dataset.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib> // rand()
#include <ctime>
typedef device_matrix<float> mat;
/*struct mapData {
	size_t phoneme;
	float* inputFeature;
};
typedef map<string, mapData> NameFtreMap;
*/

Dataset::Dataset(){
	_featureDimension=39;
	_stateDimension=0;
	_numOfTrainData=0;
	_numOfPhoneme=0;
	_trainSize=0;
	_validSize=0;
	
	_trainDataNameMatrix = NULL;
	_testDataNameMatrix = NULL;
	_trainDataMatrix = NULL;
	_testDataMatrix = NULL;
	_labelMatrix = NULL;
	_trainX = NULL;
	_validX = NULL;
	_trainY = NULL;
	_validY = NULL;
	_trainSetFlag = false;
	_validSetFlag = true;
	_batchCtr = 0;
}
Dataset::Dataset(const char* trainPath, size_t trainDataNum, const char* testPath, size_t testDataNum, const char* labelPath, size_t labelDataNum, size_t labelNum, size_t inputDim, size_t outputDim, size_t phonemeNum){
	
	_trainDataNameMatrix = NULL;
	_testDataNameMatrix = NULL;
	_trainDataMatrix = NULL;
	_testDataMatrix = NULL;
	_labelMatrix = NULL;
	_trainX = NULL;
	_validX = NULL;
	_trainY = NULL;
	_validY = NULL;
	_trainSetFlag = false;
	_validSetFlag = false;
	_batchCtr = 0;
	_numOfTrainData = trainDataNum;
	_numOfTestData = testDataNum;
	_numOfLabel = labelNum;
	_numOfPhoneme = phonemeNum;
	_featureDimension = inputDim;
	_stateDimension = outputDim;
	_frameRange = 0;
	
	size_t count  = 0, dataCount = 0;
	short split = 0;	
	_trainDataNameMatrix  = new string[trainDataNum];	
	_trainDataMatrix = new float*[trainDataNum];
	for(int i = 0;i<trainDataNum;i++){
		_trainDataMatrix[i] = new float [inputDim];
	}
	cout << "inputting train feature file:\n";	
	//NameFtreMap InputMap;
	ifstream fin(trainPath);
	if(!fin) cout<<"Can't open this file!!!\n";
	string s, tempStr;
	while(getline(fin, s) && count<trainDataNum){
		count++;

		unsigned int pos  = s.find(" ");
		unsigned int initialPos = 0;
		split=0;
		while(split<inputDim+1){
			dataCount++;
			split++;
			
			tempStr= s.substr(initialPos, pos-initialPos);
			if (split==1){
				*(_trainDataNameMatrix+count-1) = tempStr;
			}

			else{
				_trainDataMatrix[count-1][split-2] = atof(tempStr.c_str());
			}		
			initialPos = pos+1;
			pos=s.find(" ", initialPos);
		}		
	}		
	
	fin.close();	
	cout << "inputting testing file:\n";
	size_t testCount = 0, testDataCount = 0;
	 split = 0;	
	_testDataNameMatrix  = new string[testDataNum];	
	_testDataMatrix = new float*[testDataNum];
	for(int i = 0;i<testDataNum;i++){
		_testDataMatrix[i] = new float [inputDim];
	}
	
	ifstream finTest(testPath);
	if(!finTest) cout<<"Can't open this file!!!\n";
	string sTest, tempStrTest;
	while(getline(finTest, sTest)&&testCount<testDataNum){
		testCount++;
		unsigned int posTest  = sTest.find(" ");
		unsigned int initialPos = 0;
		split=0;
		while(split<inputDim+1){
			testDataCount++;
			split++;
			
			tempStrTest= sTest.substr(initialPos, posTest-initialPos);
			if (split==1){
				*(_testDataNameMatrix+testCount-1) = tempStrTest;
			}
			else{
				_testDataMatrix[testCount-1][split-2] = atof(tempStrTest.c_str());
			}		
			initialPos = posTest+1;
			posTest=sTest.find(" ", initialPos);
		}		
	}
	finTest.close();

	cout << "inputting training label file:\n";
	size_t countLabel  = 0, labelDataCount = 0, numForLabel=0;
	 split = 0;	
		
	_labelMatrix = new int[labelDataNum]; 

	ifstream finLabel(labelPath);
	if(!finLabel) cout<<"Can't open this file!!!\n";
	string sLabel, tempStrLabel, preLabel= "" ;
	while(getline(finLabel, sLabel)){
		countLabel++;

		unsigned int pos  = sLabel.find(",");
		unsigned int initialPos = 0;
		split=0;
		//string tmpName;
		while(split<2){
			labelDataCount++;
			split++;
			
			tempStrLabel = sLabel.substr(initialPos, pos-initialPos);

			if (split==2){
				if(tempStrLabel.compare(preLabel)!=0){
					if(_labelMap.find(tempStrLabel)==_labelMap.end()){
						numForLabel++;
						_labelMap.insert(pair<string, int>(tempStrLabel, numForLabel));	
					}
					preLabel = tempStrLabel;
				}

			
			*(_labelMatrix+countLabel-1)=_labelMap.find(tempStrLabel)->second;
			}
			initialPos = pos+1;
			pos=sLabel.find(",", initialPos);
		}		
	}		
	
	finLabel.close();	
	// initialize all pointers
	dataSegment(0.9);
	// put things into dataMatrix
};
Dataset::Dataset(Data data, char mode){
	
	switch(mode){
	case 'F':
	case 'f':
		{	
		//Set private members
		_numOfTrainData = data.trainDataNum;
		_numOfTestData = data.testDataNum;
		_numOfLabel = data.labelNum;
		_numOfPhoneme = data.phonemeNum;
		_featureDimension = data.inputDim;
		_stateDimension = data.outputDim;
		_frameRange = data.frameRange;
			
		size_t  dataCount = 0;
		size_t count = 0;
		size_t split = 0;
		string* tempTrainDataNameMatrix = new string[data.trainDataNum];
		float** tempTrainDataMatrix = new float*[data.trainDataNum];		
		for(int i=0;i<data.trainDataNum;i++){
			tempTrainDataMatrix[i] = new float [data.inputDim];
		}
		
		ifstream fin(data.trainPath);
		if(!fin) cout<<"Can't open the train data!\n";
		else cout<<"Inputting train data!\n";
		string s, tempStr;
		while(getline(fin,s)&&count<data.trainDataNum){
			count++;
			size_t pos = s.find(" ");
			size_t initialPos=0;
			split=0;
			string tmpName;
			while(split<data.inputDim+1){
				dataCount++;
				split++;
				
				tempStr= s.substr(initialPos, pos-initialPos);
				if (split==1){
					*(tempTrainDataNameMatrix+count-1) = tempStr;
				}

				else{
					tempTrainDataMatrix[count-1][split-2] = atof(tempStr.c_str());
				}		
				initialPos = pos+1;
				pos=s.find(" ", initialPos);
			}		
			
		}	
	
		_trainDataNameMatrix = new string[data.trainDataNum];
		_trainDataMatrix = new float*[data.trainDataNum];
		
		for(int i=0;i<data.trainDataNum;i++){
			_trainDataMatrix[i]=new float[data.inputDim*(2*_frameRange+1)];
		}
		for(int i=0;i<data.trainDataNum;i++){
			unsigned int pos = (*(tempTrainDataNameMatrix+i)).find_last_of("_");				
			_trainDataNameMatrix[i]=tempTrainDataNameMatrix[i];
			string str = _trainDataNameMatrix[i].substr(0,pos);
			unsigned int num = atoi(_trainDataNameMatrix[i].substr(pos+1).c_str());
			for(int j =(_frameRange*(-1));j<=_frameRange;j++){
				int k = j;
				if(num+j<1||(i+j)>=_numOfTrainData){
					k=0;	
				}
				else {
					unsigned int pos2 = tempTrainDataNameMatrix[i+j].find_last_of("_");
					unsigned int num2 = atoi(tempTrainDataNameMatrix[i+j].substr(pos2+1).c_str());	
				
					if(num2!=(num+j))	k=0;
				}	
				
	
				for(int l=0;l<_featureDimension;l++){
				_trainDataMatrix[i][_featureDimension*(j+_frameRange)+l]=tempTrainDataMatrix[i+k][l];	
				}
			}
			
		}
		
		
		fin.close();

		 dataCount = 0;
		 count = 0;
		 split = 0;
		string* tempTestDataNameMatrix = new string[data.testDataNum];
		float** tempTestDataMatrix = new float*[data.testDataNum];		
		for(int i=0;i<data.testDataNum;i++){
			tempTestDataMatrix[i] = new float [data.inputDim];
		}
		
		ifstream finTest(data.testPath);
		if(!finTest) cout<<"Can't open the test data!\n";
		else cout<<"Inputting test data!\n";
		while(getline(finTest,s)&&count<data.testDataNum){
			count++;
			size_t pos = s.find(" ");
			 size_t initialPos=0;
			split=0;
			string tmpName;
			while(split<data.inputDim+1){
				dataCount++;
				split++;
				
				tempStr= s.substr(initialPos, pos-initialPos);
				if (split==1){
					*(tempTestDataNameMatrix+count-1) = tempStr;
				}

				else{
					tempTestDataMatrix[count-1][split-2] = atof(tempStr.c_str());
				}		
				initialPos = pos+1;
				pos=s.find(" ", initialPos);
			}		
			
		}	
	
		_testDataNameMatrix = new string[data.testDataNum];
		_testDataMatrix = new float*[data.testDataNum];
		
		for(int i=0;i<data.testDataNum;i++){
			_testDataMatrix[i]=new float[data.inputDim*(2*_frameRange+1)];
		}
		for(int i=0;i<data.testDataNum;i++){
			unsigned int pos = (*(tempTestDataNameMatrix+i)).find_last_of("_");				
			_testDataNameMatrix[i]=tempTestDataNameMatrix[i];
			string str = _testDataNameMatrix[i].substr(0,pos);
			unsigned int num = atoi(_testDataNameMatrix[i].substr(pos+1).c_str());
			for(int j =(_frameRange*(-1));j<=_frameRange;j++){
				int k = j;
				if(num+j<1||(i+j)>=_numOfTestData){
					k=0;	
				}
				else {
					unsigned int pos2 = tempTestDataNameMatrix[i+j].find_last_of("_");
					unsigned int num2 = atoi(tempTestDataNameMatrix[i+j].substr(pos2+1).c_str());	
				
					if(num2!=(num+j))	k=0;
				}	
				
	
				for(int l=0;l<_featureDimension;l++){
				_testDataMatrix[i][_featureDimension*(j+_frameRange)+l]=tempTestDataMatrix[i+k][l];	
				}
			}
		}
		finTest.close();
		
		cout << "inputting training label file:\n";
		size_t countLabel  = 0, labelDataCount = 0, numForLabel=0;
		 split = 0;	
		
		_labelMatrix = new int[data.labelDataNum]; 

		ifstream finLabel(data.labelPath);
		if(!finLabel) cout<<"Can't open this file!!!\n";
		string sLabel, tempStrLabel, preLabel= "" ;
		while(getline(finLabel, sLabel)){
			countLabel++;

			unsigned int pos  = sLabel.find(",");
			unsigned int initialPos = 0;
			split=0;
			string tmpName;
			while(split<2){
				labelDataCount++;
				split++;
			
				tempStrLabel = sLabel.substr(initialPos, pos-initialPos);
				if (split == 1) tmpName = tempStrLabel;

				if (split==2){
				if(tempStrLabel.compare(preLabel)!=0){
					if(_labelMap.find(tempStrLabel)==_labelMap.end()){
					numForLabel++;
					_labelMap.insert(pair<string, int>(tempStrLabel, numForLabel));	
					}
					preLabel = tempStrLabel;
				}

			
				*(_labelMatrix+countLabel-1)=_labelMap.find(tempStrLabel)->second;
			}
			initialPos = pos+1;
			pos=sLabel.find(",", initialPos);
		}		
	}		
	
	finLabel.close();	
			
		
		//destructor
		if(_numOfTrainData!=0) delete [] tempTrainDataNameMatrix;
		if(tempTrainDataMatrix!=NULL){
			for(int i =0;i<_numOfTrainData;i++)
				delete tempTrainDataMatrix[i];
		}
		if(_featureDimension!=0) delete []tempTrainDataMatrix;
	
		if(_numOfTestData!=0) delete [] tempTestDataNameMatrix;
		if(tempTestDataMatrix!=NULL){
			for(int i =0;i<_numOfTestData;i++)
				delete tempTestDataMatrix[i];
		}
		if(_featureDimension!=0) delete []tempTestDataMatrix;
		break;
		}
	default:
		cout<<"No match mode!"<<endl;
		break;
	}
};
Dataset::Dataset(const Dataset& data){};
Dataset::~Dataset(){
	if(_numOfTrainData!=0)
		delete [] _trainDataNameMatrix;
	if(_trainDataMatrix != NULL){
		for(int i =0 ; i<_numOfTrainData;i++)
			delete _trainDataMatrix[i];
	}
	if(_numOfPhoneme!=0)
		delete [] _trainDataMatrix;
	
	if(_testDataMatrix != NULL){
		for (int i = 0;i<_numOfTestData;i++){
			delete[] _testDataMatrix[i];
		}
	}
	if(_testDataMatrix != NULL){
		delete []_testDataMatrix;
	}
	if(_testDataNameMatrix != NULL){
		delete []_testDataNameMatrix;
	}
	if(_labelMatrix != NULL){
		delete []_labelMatrix;
	}
	if (_trainX != NULL){
		delete[] _trainX;
	}
	if (_validX != NULL){
		delete[] _validX;
	}
	delete[] _trainY;
	delete[] _validY;

};

void Dataset::saveCSV(vector<size_t> testResult){
	
	string name, phoneme;
	ofstream fout("Prediction.csv");
	if(!fout){
		cout<<"Can't write the file!"<<endl;
	}
	fout<<"Id,Prediction\n";
	cout<<testResult.size()<<endl;
	for(size_t i = 0;i<testResult.size();i++){
		name = *(_testDataNameMatrix+i);
		fout<<name<<",";
		for(map<string,int>::iterator it = _labelMap.begin();it!=_labelMap.end();it++){
			if(it->second==testResult.at(i)){
				phoneme = it->first;
				break;
			}
		}
			phoneme = _To39PhonemeMap.find(phoneme)->second;

		fout<<phoneme<<endl;
	
	}	
	fout.close();
}

//Get function
mat Dataset::getTestSet(){
	return inputFtreToMat(_testDataMatrix, getInputDim(), _numOfTestData);
}
mat Dataset::getTestSet(float** testData,size_t frameRange, size_t testNum){
	cout<<"mat row: "<<getInputDim()*(2*frameRange+1)<<endl;	
	return inputFtreToMat(testData, getInputDim()*(2*frameRange+1), testNum);

}

size_t Dataset::getNumOfTrainData(){ return _numOfTrainData; }
size_t Dataset::getNumOfTestData(){return _numOfTestData;}
size_t Dataset::getNumOfLabel(){return _numOfLabel;}
size_t Dataset::getNumOfPhoneme(){return _numOfPhoneme;}
size_t Dataset::getInputDim(){ return _featureDimension; }
size_t Dataset::getOutputDim(){return _stateDimension;}
int    Dataset::getTrainSize(){return _trainSize;}
int    Dataset::getValidSize(){return _validSize;}

string* Dataset::getTrainDataNameMatrix(){return _trainDataNameMatrix;}
string* Dataset::getTestDataNameMatrix(){return _testDataNameMatrix;}
float** Dataset::getTrainDataMatrix(){return _trainDataMatrix;}
float** Dataset::getTestDataMatrix(){return _testDataMatrix;}
map<string, int> Dataset::getLabelMap(){return _labelMap;}
map<string, string> Dataset::getTo39PhonemeMap(){return _To39PhonemeMap;}

//Load function
void Dataset::loadTo39PhonemeMap(const char* mapFilePath){
	ifstream fin(mapFilePath);
	if(!fin) cout<<"Can't open the file!\n";
	string s, sKey, sVal;//For map
	while(getline(fin, s)){
		 int pos = 0;
		 int initialPos = 0;
		int judge = 1;
		while(pos!=string::npos){
				
			pos = s.find("\t", initialPos);
			if(judge==1) sKey = s.substr(initialPos, pos-initialPos);
			else
			{
				sVal = s.substr(initialPos, pos-initialPos);
		//		cout<<sKey<<" "<<sVal<<endl;
				_To39PhonemeMap.insert(pair<string,string>(sKey,sVal));
			}
			initialPos = pos+1;
//			pos=s.find("\t", initialPos);
			judge++;
		}
	}
	fin.close();
}

//Print function
void Dataset::printTo39PhonemeMap(map<string, string> Map){
	map<string, string>::iterator MapIter;
	for(MapIter = Map.begin();MapIter!=Map.end();MapIter++){
		cout<<MapIter->first<<"\t"<<MapIter->second<<endl;	
	}
}	
void   Dataset::printLabelMap(map<string, int> Map){
	map<string, int>::iterator labelMapIter;
	for(labelMapIter = Map.begin();labelMapIter!=Map.end();labelMapIter++){
		cout<<labelMapIter->first<<" "<<labelMapIter->second<<endl;
	}
	
}
/************************************
***********MASTER WU*****************
************************************/
void Dataset::getBatch(int batchSize, mat& batch, mat& batchLabel){
	// use shuffled trainX to get batch sequentially
	float** batchFtre = new float*[batchSize];
	int*    batchOutput = new int[batchSize];
	for (int i = 0; i < batchSize; i++){
		batchFtre[i] = _trainX[ _batchCtr % _trainSize ];
		batchOutput[i] = _trainY[ _batchCtr % _trainSize ];
		_batchCtr ++;
	}

	batch = inputFtreToMat( batchFtre, getInputDim(), batchSize);
	batchLabel = outputNumtoBin( batchOutput, batchSize );
	// free tmp pointers
	delete[] batchOutput;
	delete[] batchFtre;
	batchOutput = NULL;
	batchFtre = NULL;
	// for debugging, print both matrices
}

void Dataset::getTrainSet(int trainSize, mat& trainData, vector<size_t>& trainLabel){
	if (_trainSetFlag == true){
		trainData = trainMat;
		return;
	}
	if (trainSize > _trainSize){
		cout << "requested training set size overflow, will only output "
		     << _trainSize << " training sets.\n";
		trainSize = _trainSize;
	}
	trainLabel.clear();
	// random initialize
		
	int* randIndex = new int [trainSize];
	for (int i = 0; i < trainSize; i++){
		if (trainSize == _trainSize)
			randIndex[i] = i;
		else
			randIndex[i] = rand() % _trainSize; 
	}
	float** trainFtre = new float*[trainSize];
	for (int i = 0; i < trainSize; i++){
		trainFtre[i] = _trainX[ randIndex[i] ];
		trainLabel.push_back( _trainY[ randIndex[i] ] );
	}
	trainData = inputFtreToMat(trainFtre, getInputDim(), trainSize);
	
	_trainSetFlag = true;
	trainMat = trainData;
	delete[] randIndex;
	delete[] trainFtre;
	randIndex = NULL;
	trainFtre = NULL;
}

void Dataset::getValidSet(int validSize, mat& validData, vector<size_t>& validLabel){
	if (_validSetFlag == true){
		validData = validMat;
		return;
	}
	if (validSize > _validSize){
		cout << "requested valid set size is too big, can only feed in " << _validSize << " data.\n";
	validSize = _validSize;
	}
	validLabel.clear();
	// random choose index
	cout << "validate size is : " << validSize << " " << _validSize << endl;
	int* randIndex = new int [validSize];
	for (int i = 0; i < validSize; i++){
		if (validSize == _validSize)
			randIndex[i] = i;
		else
			randIndex[i] = rand() % _validSize; 
	}
	float** validFtre = new float*[validSize];
	for (int i = 0; i < validSize; i++){
		validFtre[i] = _validX[ randIndex[i] ];
		validLabel.push_back( _validY[ randIndex[i] ] );
	}
	validData = inputFtreToMat(validFtre, getInputDim(), validSize);
	
	_validSetFlag = true;
	validMat = validData;
	delete[] validFtre;
	delete[] randIndex;
	validFtre = NULL;
	randIndex = NULL;
}




void Dataset::dataSegment( float trainProp ){
	if (_trainX != NULL){
		delete _trainX;
		delete _trainY;
		delete _validX;
		delete _validY;
	}
	cout << "start data segmenting:\n";
	cout << "num of data is "<< getNumOfTrainData() << endl;
	// segment data into training and validating set
	_trainSize = trainProp*getNumOfTrainData();
	_validSize = getNumOfTrainData() - _trainSize;
	
	//create random permutation
	vector<int> randIndex;
	
	for (int i = 0; i < getNumOfTrainData(); i++){
		randIndex.push_back( i );
	}
	random_shuffle(randIndex.begin(), randIndex.end());
	// print shuffled data
	cout << "start shuffling:\n";
	/*
	for (int i = 0; i < getNumOfTrainData(); i++){
		cout << randIndex[i] <<" ";
	}
	*/
	// 
	cout << "put feature into training set\n";
	cout << "trainingsize = " << _trainSize <<endl;
	_trainX = new float*[_trainSize];
	_trainY = new int[_trainSize];
	for (int i = 0; i < _trainSize; i++){
		_trainX[i] = _trainDataMatrix[ randIndex[i] ]; 
		_trainY[i] = _labelMatrix[ randIndex[i] ];  // depends on ahpan
	}
	cout << "put feature into validating set\n";
	cout << "validatingsize = " << _validSize <<endl;
	_validX = new float*[_validSize];
	_validY = new int[_validSize];
	for (int i = 0; i < _validSize; i++){
		_validX[i] = _trainDataMatrix [ randIndex[_trainSize + i] ];
		_validY[i] = _labelMatrix[ randIndex[_trainSize + i] ];
	}
	// debugging, print out train x y valid x y
}
mat Dataset::outputNumtoBin(int* outputVector, int vectorSize)
{
	float* tmpVector = new float[ vectorSize * _numOfLabel ];
	for (int i = 0; i < vectorSize; i++){
		for (int j = 0; j < _numOfLabel; j++){
			*(tmpVector + i*_numOfLabel + j) = (outputVector[i] == j)?1:0;
		}
	}

	mat outputMat(tmpVector, _numOfLabel, vectorSize);
	delete[] tmpVector;
	tmpVector = NULL;
	return outputMat;
}
mat Dataset::inputFtreToMat(float** input, int r, int c){
	// r shall be the number of Labels
	// c shall be the number of data
	float* inputReshaped = new float[r * c];
	for (int i = 0; i < c; i++){
		for (int j = 0; j < r; j++){
			//*(inputReshaped + i*r + j) = *(*(input + i) +j);
			*(inputReshaped + i*r + j) = input[i][j];
		}
	}
	mat outputMat(inputReshaped, r, c);
	delete[] inputReshaped;
	inputReshaped = NULL;
	return outputMat;
}
void Dataset::prtPointer(float** input, int r, int c){
	//cout << "this prints the pointer of size: " << r << " " << c << endl;
	for (int i = 0; i < c; i++){
		cout << i << endl;
		for(int j = 0; j < r; j++){
			cout <<input[i][j]<<" ";
			if ((j+1)%5 == 0) cout <<endl;
		}
		cout <<endl;
	}
	return;
}


