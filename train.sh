#DATA=/home/ahpan/DeepLearningHW1/MachineLearningDNNCourse/Data/MLDS_HW1_RELEASE_v1/
DATA=/home/hui/model/
LARRY_DATA=/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/
TYPE=fbank/
TRAIN=${LARRY_DATA}train_351.ark
TEST=${LARRY_DATA}test_351.ark
LABEL=${DATA}label.ark
INDIM=351
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.001
BSIZE=256
MAXEPOCH=100000
DECAY=0.98
DIM=${INDIM}-256-${OUTDIM}

mkdir -p model

# **********************************
# *    FOR TRAINING FROM SCRATCH   * 
# **********************************
./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName model/out.mdl --decay ${DECAY} --range 1.5 --dim ${DIM}

# **********************************
# *     LOAD MODEL AND TRAIN       *
# **********************************
#./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --outName model/out.mdl --decay ${DECAY} --dim ${DIM} --load model/load.mdl

echo "experiment done!"
