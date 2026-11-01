CC     = gcc
CFLAGS = -O3 -march=native -Wall

all: jacobi jacobi_omp

# serial version (the OpenMP pragmas are ignored)
jacobi: jacobi.c
	$(CC) $(CFLAGS) -Wno-unknown-pragmas $< -o $@

# parallel version with OpenMP
jacobi_omp: jacobi.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@

clean:
	rm -f jacobi jacobi_omp

.PHONY: all clean
