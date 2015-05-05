#DATA=/home/ahpan/DeepLearningHW1/MachineLearningDNNCourse/Data/MLDS_HW1_RELEASE_v1/
DATA=/home/hui/project/model/
TYPE=fbank/
TRAIN=${DATA}${TYPE}train.ark
TEST=${DATA}${TYPE}test.ark
LABEL=${DATA}label/label.ark
INDIM=69
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.002
BSIZE=256
MAXEPOCH=10000
DECAY=1
DIM=${INDIM}-128-${OUTDIM}

mkdir -p model

./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName model/out.mdl --decay ${DECAY} --range 1.5 --dim ${DIM}


echo "experiment done!"
