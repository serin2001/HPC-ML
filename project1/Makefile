CC      = gcc
NVCC    = nvcc
CFLAGS  = -O3 -march=native -Wall
NVFLAGS = -O3 -arch=sm_70 # sm_70 steht für Nvidia V100

# Standardmäßig ist CUDA aus (lokal). Auf dem Cluster mit 'make cuda=1' aktivieren.
CUDA ?= 0

ifeq ($(CUDA), 1)
    CFLAGS  += -DUSE_CUDA
    LDFLAGS += -lcudart
    GPU_OBJ  = jacobi_gpu.o
else
    GPU_OBJ  =
endif

.PHONY: all clean

all: jacobi_cpu jacobi_cpu_omp

# Seriell (OpenMP Pragmas werden ignoriert)
jacobi_cpu: main.c jacobi_cpu.c $(GPU_OBJ)
	$(CC) $(CFLAGS) -Wno-unknown-pragmas $^ -o $@ $(LDFLAGS) -lm

# Parallel mit OpenMP auf der CPU
jacobi_cpu_omp: main.c jacobi_cpu.c $(GPU_OBJ)
	$(CC) $(CFLAGS) -fopenmp $^ -o $@ $(LDFLAGS) -lm

# Kompiliert den CUDA-Teil nur, wenn cuda=1 gesetzt ist
jacobi_gpu.o: jacobi_gpu.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f jacobi_cpu jacobi_cpu_omp *.o
