DATA=/home/hui/project/model/
TYPE=fbank/
TRAIN=${DATA}train/gender_norm.ark
TEST=${DATA}test/test_gender_norm.ark
LABEL=${DATA}label/label.ark
INDIM=69
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.005
BSIZE=256
MAXEPOCH=10000000
MOMENTUM="0 0.3 0.6 0.9"
DECAYSET="1 0.9 0.81 0.72"
DECAY=0.999
INITMODEL=model/momentExpInit.mdl
MODELDIR=model/initexp
UNIRANGE="0.1 0.5 1 2";
DIM=${INDIM}-128-${OUTDIM}
OUTMODEL=model/${DIM}.mdl

mkdir -p model

# *****************************************
# *********TRAINING FROM SCRATCH***********
# *****************************************

./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName model/out.mdl --decay ${DECAY} --variance 1.0 --dim ${DIM}

# ******************************************
# * THIS PART IS USED FOR LOADING DNN MODEL*
# ****************************************** 

#./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.6 --outName bug.mdl --decay ${DECAY} --dim ${DIM} --load model/dump.mdl

#mkdir -p ${MODELDIR}
#for x in $UNIRANGE
#do
#echo "calling momentum $x ..."
#./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --load ${INITMODEL} --outName ${MODELDIR}/uni${x}.mdl --decay ${DECAY} --range $x | tee ${MODELDIR}/uni${x}.log
#done

#mkdir -p /model/decexp
#for x in $DECAYSET
#do
#echo "calling momentum $x ..."
#./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --load ${INITMODEL} --outName /model/decexp/dec${x}.mdl --decay $x | tee /model/decexp/dec${x}.log
#done

echo "experiment done!"
