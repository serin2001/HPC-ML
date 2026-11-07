#ifndef JACOBI_H
#define JACOBI_H

#include "stencils.h"

// CPU-Funktionen
void jacobi_sweep(int Nx, int Ny, double *U, double *V);
void jacobi_sweep_blocked(int Nx, int Ny, int B, double *U, double *V);
void jacobi_sweep_generic(int Nx, int Ny, const Stencil *S, const double *U, double *V);
void jacobi_sweep_blocked_generic(int Nx, int Ny, int B, const Stencil *S, const double *U, double *V);
double compute_norm_l1(int Nx, int Ny, const Stencil *S, const double *U);

double wtime(void);
double* new_array(int size);

JacobiResult solve_jacobi(
    int Nx, int Ny, 
    const Stencil *S, 
    double *U_start,       // Der vom Hauptprogramm initialisierte Startvektor (mit Seed)
    int kmax,              // Maximale Sweeps (z.B. fest auf 100 für die erste Messung)
    double epsilon         // Abbruchkriterium (z.B. 1e-4)
);


// GPU-Funktion (wird nur deklariert, wenn CUDA aktiv ist)
#ifdef USE_CUDA
void run_jacobi_gpu(int Nx, int Ny, int k, const Stencil *S, double *h_U, double *h_V);
#endif

#endif
