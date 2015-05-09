CC=gcc
CXX=g++
CPPFLAGS=-g -O2 -std=c++11
NVCC=nvcc -arch=sm_21 -w
CUDA_DIR=/usr/local/cuda/

EXECUTABLES=train
LIBCUMATDIR=tool/libcumatrix/
CUMATOBJ=$(LIBCUMATDIR)obj/device_matrix.o $(LIBCUMATDIR)obj/cuda_memory_manager.o
HEADEROBJ=obj/util.o obj/transforms.o obj/dnn.o obj/dataset.o obj/parser.o

LIBS=$(LIBCUMATDIR)lib/libcumatrix.a
# +==============================+
# +======== Phony Rules =========+
# +==============================+

.PHONY: debug all clean 

all:DIR TOOL $(EXECUTABLES)

debug: CPPFLAGS+=-g -DDEBUG 

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include/\
	 -I $(LIBCUMATDIR)include/\
	 -I $(CUDA_DIR)include/\
	 -I $(CUDA_DIR)samples/common/inc/

LD_LIBRARY=-L$(CUDA_DIR)lib64 -L$(LIBCUMATDIR)lib
LIBRARY=-lcuda -lcublas -lcudart -lcumatrix

TOOL:
	@echo "Checking library file in tool/libcumatrix"
	@cd tool/libcumatrix/ ;  make ; cd ../..
DIR:
	@echo "checking object and executable directory..."
	@mkdir -p obj
	@mkdir -p bin

train:$(HEADEROBJ) example/train.cpp
	@echo "compiling train.app for DNN Training"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)

jason: obj/dataset.o example/debugData.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/debugData.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY) 
CSV: $(HEADEROBJ) CSVTest.cpp 
	@echo "compiling CSV2.app for generating CSV format testing results"
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/CSV2.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY) 

clean:
	@echo "All objects and executables removed"
	@rm -f $(EXECUTABLES) obj/* ./*.app

ctags:
	@rm -f src/tags tags
	@echo "Tagging src directory"
	@cd src; ctags -a ./* ../include/*.h ; cd ..
	@echo "Tagging example directory"
	@cd example; ctags -a ./* ../include/*.h ../src/* ; cd ..
	@echo "Tagging main directory"
	@ctags -a example/* src/* include/*.h ./*
	
# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: src/%.cpp include/%.h
	@echo "compiling OBJ: $@ " 
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

obj/datasetJason.o: src/datasetJason.cpp include/dataset.h 
	@echo "compiling OBJ: $@ "
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
obj/%.o: %.cu
	@echo "compiling OBJ: $@ "
	@$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<
