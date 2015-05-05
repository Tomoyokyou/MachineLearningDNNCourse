DATA=/home/hui/project/model/
TYPE=fbank/
TRAIN=${DATA}${TYPE}train.ark
TEST=${DATA}${TYPE}test.ark
LABEL=${DATA}label/label.ark
PHONEMAP=${DATA}phones/48_39.map
INDIM=69
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
BSIZE=1000
MODELFILE=model/out.mdl
CSVFILE=result/out.csv

mkdir -p result

./bin/predict.app ${TRAIN} ${TEST} ${LABEL} ${MODELFILE} ${PHONEMAP} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
 --inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --outName ${CSVFILE}
