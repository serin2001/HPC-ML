/*
 * Project 1, GPU part: Jacobi sweep for a general 3x3 stencil with CUDA.
 * HPC for ML, WS 2026/27
 *
 * Same storage as on the CPU: u_{i,j} = U[j*(Nx+2) + i], i = 0..Nx+1,
 * j = 0..Ny+1, boundary included. One sweep:
 *     V_{i,j} = U_{i,j} - (1/sc) (S U)_{i,j}   for all interior points.
 *
 * Mapping: one thread per interior point. threadIdx.x / blockIdx.x run in
 * i-direction (contiguous in memory), so neighbouring threads of a warp
 * access neighbouring addresses (coalesced). The number of blocks is
 * rounded up, threads outside the grid do nothing -> arbitrary Nx, Ny.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hip/hip_runtime.h>

/* this file always needs the GPU declarations from jacobi.h */
#ifndef USE_CUDA
#define USE_CUDA
#endif

/* main.c and jacobi_cpu.c are C, this file is C++ (nvcc) -> C linkage */
extern "C" {
#include "jacobi.h"
}

/* block size: 32 threads in i-direction (one warp per row), 8 rows */
#define BLOCK_X 32
#define BLOCK_Y 8

/* check every CUDA call; kernel errors are otherwise silent */
/* check every GPU/HIP call; kernel errors are otherwise silent */
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        hipError_t err_ = (call);                                           \
        if (err_ != hipSuccess) {                                           \
            printf("HIP/CUDA error %s in %s, line %d\n",                     \
                   hipGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                         \
        }                                                                    \
    } while (0)
#else
/* Fallback, falls eine CPU-Zieldatei dieses Makro unabsichtlich sieht */
#define CUDA_CHECK(call) (call)
#endif


/* stencil weights in constant memory: all threads read the same 9 values,
 * which are broadcast from the constant cache. Same orientation as on the
 * CPU: c_S[0..2] = row j+1, c_S[3..5] = row j, c_S[6..8] = row j-1. */
__constant__ double c_S[9];
__constant__ double c_inv_sc;

/* ------------------------------------------------------------------------ */
/* kernel: one sweep                                                        */
/* ------------------------------------------------------------------------ */

// Der GPU-Kernel für die Triad-Operation
__global__ void triad_kernel(int n, const double* B, const double* C, double* A, double s)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        A[k] = B[k] + s * C[k];
    }
}

// Host-Schnittstelle zur Zeitmessung (wtime() muss in der main oder hier bekannt sein)
extern "C" double measure_bandwidth_gpu(int n)
{
    size_t bytes = (size_t)n * sizeof(double);
    double *d_A, *d_B, *d_C;
    double s = 2.0;

    // 1. Speicher auf der GPU allozieren (gesichert durch Makro)
    CUDA_CHECK(hipMalloc(&d_A, bytes));
    CUDA_CHECK(hipMalloc(&d_B, bytes));
    CUDA_CHECK(hipMalloc(&d_C, bytes));

    // Temp-Arrays zur Initialisierung auf dem Host
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);
    for (int k = 0; k < n; k++) {
        h_B[k] = 1.0; 
        h_C[k] = 0.5;
    }

    // Daten einmalig vor dem Benchmark auf die GPU schieben (gesichert)
    CUDA_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice));
    free(h_B); free(h_C);

    // Kernel-Konfiguration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    double tmin = 1e30;

    // 10 Benchmark-Durchläufe auf der GPU
    for (int r = 0; r < 10; r++) {
        CUDA_CHECK(hipDeviceSynchronize()); // Synchronisation gesichert
        double t0 = wtime();

        triad_kernel<<<gridSize, blockSize>>>(n, d_B, d_C, d_A, s);

        CUDA_CHECK(hipDeviceSynchronize()); // Synchronisation gesichert
        double t = wtime() - t0;
        
        if (t < tmin) tmin = t;
    }

    // Validierung: Ein Element zurückholen und prüfen
    double check_val = 0.0;
    CUDA_CHECK(hipMemcpy(&check_val, &d_A[n / 2], sizeof(double), hipMemcpyDeviceToHost));
    if (check_val != 2.0) printf("Error in the GPU triad\n");

    // Speicher freigeben (gesichert)
    CUDA_CHECK(hipFree(d_A)); 
    CUDA_CHECK(hipFree(d_B)); 
    CUDA_CHECK(hipFree(d_C));

    // Rückgabe der Bandbreite in GB/s
    return 32.0 * n / tmin * 1e-9;
}


__global__ void jacobi_sweep_kernel(int Nx, int Ny,
                                    const double * __restrict__ U,
                                    double * __restrict__ V)
{
    /* +1: interior points start at index 1 */
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i > Nx || j > Ny)          /* surplus threads of the last blocks */
        return;

    int L = Nx + 2;
    int idx = j * L + i;

    double SU = c_S[0] * U[idx + L - 1] + c_S[1] * U[idx + L] + c_S[2] * U[idx + L + 1]
              + c_S[3] * U[idx     - 1] + c_S[4] * U[idx    ] + c_S[5] * U[idx     + 1]
              + c_S[6] * U[idx - L - 1] + c_S[7] * U[idx - L] + c_S[8] * U[idx - L + 1];

    V[idx] = U[idx] - c_inv_sc * SU;
}

/* ------------------------------------------------------------------------ */
/* host helpers                                                             */
/* ------------------------------------------------------------------------ */

/* copy the stencil into constant memory (once per stencil) */
static void gpu_set_stencil(const Stencil *S)
{
    double w[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            w[3 * r + c] = S->matrix[r][c];
    double inv_sc = 1.0 / S->matrix[1][1];
    CUDA_CHECK(hipMemcpyToSymbol(c_S, w, sizeof(w)));
    CUDA_CHECK(hipMemcpyToSymbol(c_inv_sc, &inv_sc, sizeof(double)));
}

/* start one sweep on the GPU (d_U, d_V are device pointers) */
static void gpu_launch_sweep(int Nx, int Ny, const double *d_U, double *d_V)
{
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((Nx + BLOCK_X - 1) / BLOCK_X,      /* rounded up */
              (Ny + BLOCK_Y - 1) / BLOCK_Y);
    jacobi_sweep_kernel<<<grid, block>>>(Nx, Ny, d_U, d_V);
    CUDA_CHECK(hipGetLastError());
}

/* ------------------------------------------------------------------------ */
/* one sweep V = J(U), including the transfers                              */
/* ------------------------------------------------------------------------ */

/* U, V: host arrays. The boundary values of V are not changed. */
void jacobi_sweep_gpu(int Nx, int Ny, const Stencil *S, const double *U, double *V)
{
    size_t bytes = (size_t) (Nx + 2) * (Ny + 2) * sizeof(double);
    double *d_U, *d_V;

    gpu_set_stencil(S);
    CUDA_CHECK(hipMalloc(&d_U, bytes));
    CUDA_CHECK(hipMalloc(&d_V, bytes));
    CUDA_CHECK(hipMemcpy(d_U, U, bytes, hipMemcpyHostToDevice));
    /* V is copied too, so that its boundary values stay unchanged */
    CUDA_CHECK(hipMemcpy(d_V, V, bytes, hipMemcpyHostToDevice));

    gpu_launch_sweep(Nx, Ny, d_U, d_V);

    CUDA_CHECK(hipMemcpy(V, d_V, bytes, hipMemcpyDeviceToHost));
    CUDA_CHECK(hipFree(d_U));
    CUDA_CHECK(hipFree(d_V));
}

/* ------------------------------------------------------------------------ */
/* k sweeps: data stays on the GPU, pointer swap on the device              */
/* ------------------------------------------------------------------------ */

/* U_start: initial values incl. boundary; on return the result (also for
 * odd k). Times in seconds: transfer to the GPU, k sweeps, transfer back. */
GpuTimes solve_jacobi_gpu(int Nx, int Ny, const Stencil *S, double *U_start, int k)
{
    size_t bytes = (size_t) (Nx + 2) * (Ny + 2) * sizeof(double);
    double *d_U, *d_V;
    GpuTimes times;

    gpu_set_stencil(S);
    CUDA_CHECK(hipMalloc(&d_U, bytes));
    CUDA_CHECK(hipMalloc(&d_V, bytes));

    /* transfer to the GPU: both arrays need the boundary values */
    double t0 = wtime();
    CUDA_CHECK(hipMemcpy(d_U, U_start, bytes, hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_V, d_U, bytes, hipMemcpyDeviceToDevice));
    CUDA_CHECK(hipDeviceSynchronize());
    times.t_to_gpu = wtime() - t0;

    /* k sweeps; kernel starts are asynchronous -> synchronize before
     * stopping the clock */
    t0 = wtime();
    for (int s = 0; s < k; s++) {
        gpu_launch_sweep(Nx, Ny, d_U, d_V);
        double *tmp = d_U;          /* swap the pointers, not the arrays */
        d_U = d_V;
        d_V = tmp;
    }
    CUDA_CHECK(hipDeviceSynchronize());
    times.t_sweeps = wtime() - t0;

    /* after the last swap d_U points to the result */
    t0 = wtime();
    CUDA_CHECK(hipMemcpy(U_start, d_U, bytes, hipMemcpyDeviceToHost));
    times.t_to_cpu = wtime() - t0;

    CUDA_CHECK(hipFree(d_U));
    CUDA_CHECK(hipFree(d_V));
    return times;
}

/* ------------------------------------------------------------------------ */
/* tests                                                                    */
/* ------------------------------------------------------------------------ */

/* test of Exercise 2: u = x^2 + y^2 on all grid points, one sweep gives
 * exactly V = U + test_factor * h^2 (test_factor = 4/sc) */
void check_sweep_gpu(int Nx, int Ny, const Stencil *S)
{
    int L = Nx + 2;
    double h = 1.0 / (Nx + 1);
    double *U = new_array((Nx + 2) * (Ny + 2));
    double *V = new_array((Nx + 2) * (Ny + 2));

    for (int j = 0; j <= Ny + 1; j++)
        for (int i = 0; i <= Nx + 1; i++) {
            double x = i * h, y = j * h;
            U[j * L + i] = x * x + y * y;
        }

    jacobi_sweep_gpu(Nx, Ny, S, U, V);

    double err = 0.0;
    double expected = S->test_factor * h * h;
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++) {
            double d = fabs(V[j * L + i] - U[j * L + i] - expected);
            if (d > err) err = d;
        }
    /* the boundary of V must still be 0 (new_array) */
    int boundary_ok = 1;
    for (int i = 0; i <= Nx + 1; i++)
        if (V[i] != 0.0 || V[(Ny + 1) * L + i] != 0.0) boundary_ok = 0;
    for (int j = 0; j <= Ny + 1; j++)
        if (V[j * L] != 0.0 || V[j * L + Nx + 1] != 0.0) boundary_ok = 0;

    printf("  GPU %s, Nx = %d, Ny = %d: err/h^2 = %.1e, boundary %s  %s\n",
           S->name, Nx, Ny, err / (h * h), boundary_ok ? "unchanged" : "CHANGED",
           (err / (h * h) < 1e-6 && boundary_ok) ? "OK" : "WRONG");
    free(U);
    free(V);
}

/* comparison with the CPU code: same random initial values, k sweeps */
void compare_cpu_gpu(int Nx, int Ny, const Stencil *S, int k)
{
    int len = (Nx + 2) * (Ny + 2);
    double *U_cpu = new_array(len);
    double *U_gpu = new_array(len);

    srand(7);
    for (int q = 0; q < len; q++)
        U_cpu[q] = U_gpu[q] = (double) rand() / RAND_MAX;

    solve_jacobi(Nx, Ny, S, U_cpu, k, 0.0, 0);   /* exactly k sweeps */
    solve_jacobi_gpu(Nx, Ny, S, U_gpu, k);

    double diff = 0.0, nrm = 0.0;
    for (int q = 0; q < len; q++) {
        double d = fabs(U_cpu[q] - U_gpu[q]);
        if (d > diff) diff = d;
        if (fabs(U_cpu[q]) > nrm) nrm = fabs(U_cpu[q]);
    }
    /* not necessarily bitwise equal: the GPU may use FMA instructions */
    printf("  CPU vs GPU, %s, %d x %d, k = %d: max |diff| / max |u| = %.1e  %s\n",
           S->name, Nx, Ny, k, diff / nrm, (diff / nrm < 1e-12) ? "OK" : "WRONG");
    free(U_cpu);
    free(U_gpu);
}
