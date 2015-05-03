DATA=/home/hui/model/
DATA_LARRY=/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/
TYPE=fbank/
TRAIN=${DATA}train/train_gender_norm.svm
#TRAIN=${DATA_LARRY}train_351.ark
TEST=${DATA}test/test_gender_norm.svm
#TEST=${DATA_LARRY}test_351.ark
LABEL=${DATA}label.ark
INDIM=69
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.005
BSIZE=256
MAXEPOCH=500
MOMENTUM="0 0.3 0.6 0.9"
DECAYSET="1 0.9 0.81 0.72"
DECAY=0.99999
INITMODEL=model/momentExpInit.mdl
MODELDIR=model/initexp
UNIRANGE="0.1 0.5 1 2";
DIM=${INDIM}-128-64-${OUTDIM}
OUTMODEL=model/${DIM}.mdl

mkdir -p model

# *****************************************
# *********TRAINING FROM SCRATCH***********
# *****************************************

./bin/train.app ${TRAIN} ${TEST} ${LABEL} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName model/out.mdl --decay ${DECAY} --variance 1.0 --dim ${DIM}

# ******************************************
# * THIS PART IS USED FOR LOADING DNN MODEL*
# ****************************************** 

#./bin/train.app ${TRAIN} ${TEST} ${LABEL} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.6 --outName bug.mdl --decay ${DECAY} --dim ${DIM} --load model/dump.mdl


echo "experiment done!"
