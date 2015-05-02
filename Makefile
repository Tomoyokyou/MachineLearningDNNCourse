CC=gcc
CXX=g++
CPPFLAGS=-g -O2 -std=c++11 $(INCLUDE)
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

all:DIR $(EXECUTABLES)

$(LIBCUMATDIR)lib/libcumatrix.a:
	@echo "Missing library file, trying to fix it in tool/libcumatrix"
	@cd tool/libcumatrix/ ; make ; cd ../..
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

DIR:
	@echo "checking object and executable directory..."
	@mkdir -p obj
	@mkdir -p bin

train:  $(HEADEROBJ) example/train.cpp
	@echo "compiling train.app for DNN Training"
	@$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/$@.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY)


CSV: $(HEADEROBJ) CSVTest.cpp 
	@echo "compiling CSV2.app for generating CSV format testing results"
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o bin/CSV2.app $^ $(LIBS) $(LIBRARY) $(LD_LIBRARY) 

clean:
	@echo "All objects and executables removed"
	@rm -f $(EXECUTABLES) obj/* ./*.app

ctags:
	@rm -f src/tags tags
	@echo "Tagging src directory"
	@cd src; ctags -a *.cpp ../include/*.h; ctags -a *.cu ../include/*.h; cd ..
	@echo "Tagging main directory"
	@ctags -a *.cpp src/* ; ctags -a *.cu src/*
	
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
