#include<iostream>
#include "dataset.h"

using namespace std;

int main(){
	const char* trainPath = "/home/hui/model/train/train_gender_norm.svm";
	Dataset allData(trainPath);
	Dataset trainData;
	Dataset validData;
	allData.dataSegment(trainData, validData);

	cout << trainData.getDataNum() << endl;
	cout << trainData.getFeatureDim() << endl;
	vector<size_t> label = trainData.getLabel_vec();
	cout << "vector size is: " << label.size();
	mat label_mat = trainData.getLabel_mat();
	//label_mat.print();

	

}
