#ifndef JACOBI_H
#define JACOBI_H

#include "stencils.h"

/* helpers (main.c) */
double wtime(void);
double *new_array(int len);

/* sweeps (jacobi_cpu.c) */
void jacobi_sweep(int Nx, int Ny, double *U, double *V);
void jacobi_sweep_blocked(int Nx, int Ny, int B, double *U, double *V);
void jacobi_sweep_generic(int Nx, int Ny, const Stencil *S,
                          const double *U, double *V);
void jacobi_sweep_blocked_generic(int Nx, int Ny, int B, const Stencil *S,
                                  const double *U, double *V);

/* residual and solver (jacobi_cpu.c) */
double compute_norm_l1(int Nx, int Ny, const Stencil *S, const double *U);

extern double measure_bandwidth_gpu(int n);

JacobiResult solve_jacobi(int Nx, int Ny, const Stencil *S, double *U_start,
                          int kmax, double epsilon, int check_every);

/* GPU part (jacobi_gpu.cu), only when compiled with nvcc -DUSE_CUDA */
typedef struct {
    double t_to_gpu;   /* transfer CPU -> GPU in s */
    double t_sweeps;   /* k sweeps on the GPU in s */
    double t_to_cpu;   /* transfer GPU -> CPU in s */
} GpuTimes;

#ifdef USE_CUDA
void jacobi_sweep_gpu(int Nx, int Ny, const Stencil *S, const double *U, double *V);
GpuTimes solve_jacobi_gpu(int Nx, int Ny, const Stencil *S, double *U_start, int k);
void check_sweep_gpu(int Nx, int Ny, const Stencil *S);
void compare_cpu_gpu(int Nx, int Ny, const Stencil *S, int k);
#endif

#endif
