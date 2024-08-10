# compile in debug mode
# to debug with cuda-gd use -G instead of -lineinfo and also delete optimization flag -O
DEBUG_FLAG := -g -lineinfo

SMS ?= 80 86 # compile for target architecture
GENCODE_FLAGS := $(foreach sm, $(SMS), -gencode arch=compute_$(sm),code=sm_$(sm))

CC := nvcc # .h .cpp .cu -> object file
LINKER := $(CC) # .o -> executable file

CPPFLAGS := -std=c++17 -O2 -I include # used in compilation -O2
LDFLAGS := -lm -lcublas -lopenblas # used in link

# for testing different versions of matrix multiplication: TEST := tests/test_fp32 or test_fp16
# for testing cuBLAS: TEST := tests/test_cuBLAS or test_cuBLAS_cudacore_fp32 or test_cuBLAS_tensorcore_fp16
TEST := test_fp32

# cuda core matmul version: MATMUL  src/MMult_v0 or MMult_v1 or ...
# tensor core matmul version: MATMUL := src/MMult_t_v0 or MMult_t_v1 or ...
MATMUL := matmul_v7

TARGET := results/result_$(MATMUL)_$(TEST).txt

UTIL_OBJS := compare_mat.o gen_mat.o matmul_ref.o convert_mat.o transpose_mat.o

TEST_OBJS := $(TEST).o $(MATMUL).o

OBJS := $(UTIL_OBJS) $(TEST_OBJS)

.PHONY: all run clean

all:
	make bin/$(TEST).exe
	
bin/$(TEST).exe: $(OBJS) 
	$(LINKER) $^ $(LDFLAGS) -o $@

%.o: %.cpp parameters.h
	$(CC) $(CPPFLAGS) $(GENCODE_FLAGS) $(DEBUG_FLAG) -c $< -o $(@F)

%.o: %.cu parameters.h
	$(CC) $(CPPFLAGS) $(GENCODE_FLAGS) $(DEBUG_FLAG) -c $< -o $(@F)

run:
	make all
	echo "version = '$(MATMUL) + $(TEST)';" > $(TARGET)
	./bin/$(TEST).exe >> $(TARGET)

clean:
	mv *.o build/

rm:
	rm build/*.o

vpath %.cu src tests
vpath %.cpp src tests 
vpath %.h include
vpath %.o build