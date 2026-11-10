# make             -> jacobi_cpu (CPU only, gcc)
# make jacobi_gpu  -> CPU + GPU (needs nvcc, e.g. on Amandus)
CC     = gcc
NVCC   = nvcc
CFLAGS = -O3 -march=native -fopenmp

all: jacobi_cpu

jacobi_cpu: main.c jacobi_cpu.c jacobi.h stencils.h
	$(CC) $(CFLAGS) -o $@ main.c jacobi_cpu.c -lm

# the C files are compiled with gcc, the .cu file with nvcc, linked with nvcc
jacobi_gpu: main.c jacobi_cpu.c jacobi_gpu.cu jacobi.h stencils.h
	$(CC) $(CFLAGS) -DUSE_CUDA -c main.c -o main_gpu.o
	$(CC) $(CFLAGS) -c jacobi_cpu.c -o jacobi_cpu.o
	$(NVCC) -O3 -c jacobi_gpu.cu -o jacobi_gpu.o
	$(NVCC) -o $@ main_gpu.o jacobi_cpu.o jacobi_gpu.o -Xcompiler -fopenmp -lm

clean:
	rm -f jacobi_cpu jacobi_gpu *.o
