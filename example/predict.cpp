#include "parser.h"
#include "dnn.h"
#include "dataset.h"
#include "util.h"
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <device_matrix.h>
#include <map>

using namespace std;

typedef device_matrix<float> mat;
typedef vector<size_t> Seq;

void dnnPredict(DNN& dnn,Dataset& ds,vector<Seq>& result,size_t bsize);
void write(ofstream& out,const vector<Seq>& result,Dataset& ds);
string myMap(size_t lab);
void buildMap(map<size_t,string>& out,Dataset& ds);
void myUsage(){cerr<<"$cmd [inputfile] [testfile] [labelfile] [modelfile] [mapfile] --labeldim [] --phonenum [] --trainnum [] --testnum [] --labelnum [] --inputdim [] --outputdim [] "<<endl;}

int main(int argc,char** argv){
	srand((unsigned)time(NULL));
	PARSER p;
	p.addMust("trainFilename",false);
	p.addMust("testFilename",false);
	p.addMust("labelFilename",false);
	p.addMust("modelFilename",false);
	p.addMust("mapFilename",false);
	p.addOption("--labeldim",true);
	p.addOption("--phonenum",true);
	p.addOption("--trainnum",true);
	p.addOption("--testnum",true);
	p.addOption("--labelnum",true);
	p.addOption("--inputdim",true);
	p.addOption("--outputdim",true);
	p.addOption("--batchsize",true);
	p.addOption("--outName",false);
	string trainF,testF,labelF,outF,loadF,mapF;
	size_t labdim,phonenum,trainnum,testnum,labelnum,indim,outdim,b_size;
	if(!p.read(argc,argv)){
		myUsage();
		return 1;
	}
	p.getString("trainfilename",trainF);
	p.getString("testfilename",testF);
	p.getString("labelFilename",labelF);
	p.getString("modelFilename",loadF);
	p.getString("mapFilename",mapF);
	if(!p.getNum("--labeldim",labdim)){return 1;}
	if(!p.getNum("--phonenum",phonenum)){return 1;}
	if(!p.getNum("--trainnum",trainnum)){return 1;}
	if(!p.getNum("--testnum",testnum)){return 1;}
	if(!p.getNum("--labelnum",labelnum)){return 1;}
	p.getNum("--inputdim",indim);
	p.getNum("--outputdim",outdim);
	if(!p.getNum("--batchsize",b_size)){b_size=1000;}
	if(!p.getString("--outName",outF)){outF="out.csv";}
	p.print();
	Dataset dataset = Dataset(trainF.c_str(),trainnum,testF.c_str(),testnum,labelF.c_str(),labelnum,labdim,indim,outdim,phonenum);
	dataset.dataSegment(0.8);
	dataset.loadTo39PhonemeMap(mapF.c_str());
		DNN nnload;
		if(nnload.load(loadF)){
			vector<Seq>* r_ptr=new vector<Seq>;
			dnnPredict(nnload,dataset,*r_ptr,b_size);
			ofstream out(outF.c_str());
			if(!out){cerr<<"ERROR:fail opening out file"<<endl;return 1;}
			cout<<"writing file\""<<outF<<"\"...";
			write(out,*r_ptr,dataset);	
			cout<<"done!"<<endl;
			r_ptr->clear();
			delete r_ptr;
		}
		else{	cerr<<"loading file:"<<loadF<<" failed! please check again..."<<endl;return 1;}
	cout<<"end of testing!";
	return 0;
}

void dnnPredict(DNN& dnn,Dataset& ds,vector<Seq>& result,size_t bsize){
	float** _data=ds.getTestDataMatrix();
	size_t tnum=ds.getNumOfTestData(),fdim=ds.getInputDim();
	mat batch(fdim,bsize);
	Seq temp;
	float* h_data=new float[bsize*fdim];
	size_t t,k,l;
	cout<<"begin testing:"<<endl;
	for(t=0;t<tnum/bsize;++t){
		for(k=0;k<bsize;++k){
			for(l=0;l<fdim;++l)
				h_data[k*fdim+l]=_data[t*bsize+k][l];
		}
		CCE(cudaMemcpy(batch.getData(),h_data,bsize*fdim*sizeof(float),cudaMemcpyHostToDevice));
		temp.clear();
		dnn.predict(temp,batch);
		result.push_back(temp);
	}
	size_t residual=tnum-t*bsize;
	if(residual){
		batch.resize(fdim,residual);
		for(k=0;k<residual;++k){
			for(l=0;l<fdim;++l)
				h_data[k*fdim+l]=_data[t*bsize+k][l];
		}
		CCE(cudaMemcpy(batch.getData(),h_data,residual*fdim*sizeof(float),cudaMemcpyHostToDevice));
		temp.clear();
		dnn.predict(temp,batch);
		result.push_back(temp);
	}
	cout<<"total feature number: "<<t*bsize+residual<<endl;
	delete [] h_data;
}
void write(ofstream& out,const vector<Seq>& result,Dataset& ds){
	string * nameset=ds.getTestDataNameMatrix();
	size_t fnum=ds.getNumOfTestData(),acc=0,cur;
	map<size_t,string> finalMap;
	buildMap(finalMap,ds);
	map<size_t,string>::iterator it;
	out<<"Id,Prediction"<<endl;
	for(size_t t=0;t<result.size();++t){
		cur=result[t].size();
		for(size_t k=0;k<cur;++k){
			it=finalMap.find(result[t].at(k));
			if(it!=finalMap.end())
				out<<nameset[acc+k]<<","<<it->second<<endl;
			else{cerr<<"WARNING: unknown label detected!"<<endl;}
		}
		acc=acc+cur;
	}
	cout<<"  total "<<acc<<" features writen  ";
}

string myMap(size_t lab){
	switch(lab){
		case 0:return "aa";break;
		case 1:return "ae";break;
		case 2:return "ah";break;
		case 3:return "aa";break;
		case 4:return "aw";break;
		case 5:return "ah";break;
		case 6:return "ay";break;
		case 7:return "b";break;
		case 8:return "ch";break;
		case 9:return "sil";break;
		case 10:return "d";break;
		case 11:return "dh";break;
		case 12:return "dx";break;
		case 13:return "eh";break;
		case 14:return "l";break;
		case 15:return "n";break;
		case 16:return "sil";break;
		case 17:return "er";break;
		case 18:return "ey";break;
		case 19:return "f";break;
		case 20:return "g";break;
		case 21:return "hh";break;
		case 22:return "ih";break;
		case 23:return "ih";break;
		case 24:return "iy";break;
		case 25:return "jh";break;
		case 26:return "k";break;
		case 27:return "l";break;
		case 28:return "m";break;
		case 29:return "n";break;
		case 30:return "ng";break;
		case 31:return "ow";break;
		case 32:return "oy";break;
		case 33:return "p";break;
		case 34:return "r";break;
		case 35:return "s";break;
		case 36:return "sh";break;
		case 37:return "sil";break;
		case 38:return "t";break;
		case 39:return "th";break;
		case 40:return "uh";break;
		case 41:return "uw";break;
		case 42:return "v";break;
		case 43:return "sil";break;
		case 44:return "w";break;
		case 45:return "y";break;
		case 46:return "z";break;
		case 47:return "sh";break;
		default:return "sil";break;
	}
}

void buildMap(map<size_t,string>& out,Dataset& ds){
	map<string,int> firstMap=ds.getLabelMap();
	map<string,string> to39=ds.getTo39PhonemeMap();
	map<size_t,string> midMap;
	map<string,int>::iterator it1;
	map<size_t,string>::iterator itmid;
	map<string,string>::iterator it2;
	for(it1=firstMap.begin();it1!=firstMap.end();++it1){
		itmid=midMap.find((size_t)it1->second);
		if(itmid!=midMap.end()){cerr<<"Warning: double key in new map..."<<endl;exit(1);}
		midMap.insert(pair<size_t,string>((size_t)it1->second,it1->first));
	}
	for(itmid=midMap.begin();itmid!=midMap.end();++itmid){
		it2=to39.find(itmid->second);
		if(it2!=to39.end())
			out.insert(pair<size_t,string>(itmid->first,it2->second));
	}
}
