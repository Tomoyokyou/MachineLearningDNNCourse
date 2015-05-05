#include "parser.h"
#include "dnn.h"
#include "dataset.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

void myUsage(){cerr<<"$cmd [inputfile] [testfile] [labelfile] --labeldim [] --phonenum [] --trainnum [] --testnum [] --labelnum [] --inputdim [] --outputdim [] --maxEpoch [] --momentum [] --decay [] --load []"<<endl;}

int main(int argc,char** argv){
	srand((unsigned)time(NULL));
	PARSER p;
	p.addMust("trainFilename",false);
	p.addMust("testFilename",false);
	p.addMust("labelFilename",false);
	p.addOption("--labeldim",true);
	p.addOption("--phonenum",true);
	p.addOption("--trainnum",true);
	p.addOption("--testnum",true);
	p.addOption("--labelnum",true);
	p.addOption("--inputdim",true);
	p.addOption("--outputdim",true);
	p.addOption("--rate",true);
	p.addOption("--segment",true);
	p.addOption("--batchsize",true);
	p.addOption("--maxEpoch",true);
	p.addOption("--momentum",true);
	p.addOption("--outName",false);
	p.addOption("--load",false);
	p.addOption("--decay",true);
	p.addOption("--variance",true);
	p.addOption("--range",true);
	p.addOption("--dim",false);
	string trainF,testF,labelF,outF,loadF,dims;
	size_t labdim,phonenum,trainnum,testnum,labelnum,indim,outdim,b_size,m_e;
	float rate,segment,momentum,decay,var;
	Init _inittype;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("trainfilename",trainF);
	p.getString("testfilename",testF);
	p.getString("labelFilename",labelF);
	if(!p.getNum("--labeldim",labdim)){return 1;}
	if(!p.getNum("--phonenum",phonenum)){return 1;}
	if(!p.getNum("--trainnum",trainnum)){return 1;}
	if(!p.getNum("--testnum",testnum)){return 1;}
	if(!p.getNum("--labelnum",labelnum)){return 1;}
	p.getNum("--inputdim",indim);
	p.getNum("--outputdim",outdim);
	if(!p.getNum("--rate",rate)){rate=0.1;}
	if(!p.getNum("--segment",segment)){segment=0.8;}
	if(!p.getNum("--batchsize",b_size)){b_size=128;}
	if(!p.getNum("--maxEpoch",m_e)){m_e=10000;}
	if(!p.getNum("--momentum",momentum)){momentum=0;}
	if(!p.getString("--outName",outF)){outF="out.mdl";}
	if(!p.getNum("--decay",decay)){decay=1;}
	if(p.getNum("--variance",var)&&p.getNum("--range",var)){cerr<<"--variance for normal init, --range for uniform init, not both!"<<endl;return 1;}
	if(!p.getNum("--variance",var)){var=0.2;_inittype=NORMAL;}
	if(!p.getNum("--range",var)){var=1;_inittype=UNIFORM;}
	if(!p.getString("--dim",dims)){cerr<<"wrong hidden layer dimensions";return 1;}
	p.print();
	Dataset dataset = Dataset(trainF.c_str(),trainnum,testF.c_str(),testnum,labelF.c_str(),labelnum,labdim,indim,outdim,phonenum);
	dataset.dataSegment(segment);
	if(p.getString("--load",loadF)){
		DNN nnload;
		if(nnload.load(loadF)){
		nnload.setDataset(&dataset);
		nnload.setLearningRate(rate);
		nnload.setMomentum(momentum);
		nnload.train(b_size,m_e,10000,10000,decay);
		nnload.save(outF);
		}
		else{	cerr<<"loading file:"<<loadF<<" failed! please check again..."<<endl;return 1;}
	}
	else{
		vector<size_t>dim;
		parseDim(dims,dim);
		//for(int i = 0; i < dim.size(); i++){
		//	cout << dim[i] << " " ;
		//}
		cout << endl;
		DNN dnn(&dataset,rate,momentum,var,_inittype,dim,BATCH);
		dnn.train(b_size,m_e,10000,10000,decay);
		dnn.save(outF);
	}
	cout<<"end of training!";
	cout<<"\n model saved as :"<<outF<<endl;
	return 0;
}

