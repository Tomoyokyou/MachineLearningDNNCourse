DATA=/home/hui/
LARRY_DATA=/home/larry/Documents/data/MLDS_HW1_RELEASE_v1/mfcc/
TYPE=fbank/
TRAIN=${LARRY_DATA}train_351.ark
TEST=${LARRY_DATA}test_351.ark
LABEL=${DATA}model/label.ark
PHONEMAP=${DATA}feat/phones/48_39.map
INDIM=351
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
BSIZE=1000
MODELFILE=best.mdl
CSVFILE=result/out.csv

mkdir -p result

gdb --args ./bin/predict.app ${TRAIN} ${TEST} ${LABEL} ${MODELFILE} ${PHONEMAP} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
 --inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --outName ${CSVFILE}
