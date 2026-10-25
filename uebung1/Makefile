CC     = gcc
CFLAGS = -O3 -march=native -Wall

all: stencil stencil_O0 stencil_omp

# serial version (the OpenMP pragmas are ignored)
stencil: stencil.c
	$(CC) $(CFLAGS) -Wno-unknown-pragmas $< -o $@

# same program without optimization, to compare -O0 and -O3
stencil_O0: stencil.c
	$(CC) -O0 -Wall -Wno-unknown-pragmas $< -o $@

# parallel version with OpenMP
stencil_omp: stencil.c
	$(CC) $(CFLAGS) -fopenmp $< -o $@

clean:
	rm -f stencil stencil_O0 stencil_omp

.PHONY: all clean
