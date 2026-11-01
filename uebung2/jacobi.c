/*
 * Exercise 2, programming task: one Jacobi sweep for the five-point stencil,
 * layer condition and spatial blocking
 * HPC for ML, WS 2026/27
 *
 * Compile: make        (builds jacobi and jacobi_omp)
 * Run:     ./jacobi             only the version without blocking
 *          ./jacobi B           also the blocked version with block width B
 *          OMP_NUM_THREADS=4 ./jacobi_omp B
 *
 * Grid: Nx interior points in x-direction (index i), Ny interior points in
 * y-direction (index j). A grid function is stored INCLUDING the boundary
 * points in an array of length (Nx+2)*(Ny+2), row by row:
 *     u_{i,j} = U[j*(Nx+2) + i],   i = 0,...,Nx+1,  j = 0,...,Ny+1.
 * One row (fixed j) is contiguous in memory and has length Nx+2.
 * (In the lecture, the inner index is called j and the row length jmax.)
 *
 * One Jacobi sweep for -Laplace(u) = 0 computes for all interior points
 *     V_{i,j} = 1/4 ( U_{i-1,j} + U_{i+1,j} + U_{i,j-1} + U_{i,j+1} ).
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/* ------------------------------------------------------------------------ */
/* (a) one Jacobi sweep V = J(U)                                            */
/* ------------------------------------------------------------------------ */

/* The boundary values of V are not changed. */
void jacobi_sweep(int Nx, int Ny, double *U, double *V)
{
    /* TODO (a); in part (d), parallelize it with OpenMP */
    int L = Nx + 2;
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++) 
            V[j * L + i] = 0.25 * (U[(j -1)* L + i] + U[j * L + i - 1] + U[j * L + i + 1] + U[(j + 1) * L + i]);
}

/* ------------------------------------------------------------------------ */
/* (c) the same sweep with spatial blocking of the i-loop                   */
/* ------------------------------------------------------------------------ */

/* The interior columns i = 1,...,Nx are split into blocks of width B (the
 * last block may be shorter). For each block, the sweep runs over all rows j,
 * so only three rows of length B have to stay in the cache. */
void jacobi_sweep_blocked(int Nx, int Ny, int B, double *U, double *V)
{
    /* TODO (c); in part (d), parallelize it with OpenMP */
    int L = Nx + 2;    
    for (int i = 1; i <= Nx; i += B) {
        int i_end = (i + B - 1 < Nx) ? i + B - 1 : Nx;
        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= Ny; j++)        
            for (int ii = i; ii <= i_end; ii++) 
                V[j * L + ii] = 0.25 * (U[(j -1)* L + ii] + U[j * L + ii - 1] + U[j * L + ii + 1] + U[(j + 1) * L + ii]);
    }
}

/* ======================================================================== */
/* Everything below is given.                                               */
/* ======================================================================== */

/* wall-clock time in seconds (see the C introduction) */
double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* array of length len, all values set to 0 */
double *new_array(int len)
{
    double *a = (double *) malloc(len * sizeof(double));
    if (a == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (int k = 0; k < len; k++)
        a[k] = 0.0;
    return a;
}

/* Test with u(x,y) = x^2 + y^2 on all grid points (including the boundary),
 * h = 1/(Nx+1), x = ih, y = jh. Since Laplace(u) = 4, one sweep gives exactly
 *     V_{i,j} = u_{i,j} + h^2.
 * B = 0: test jacobi_sweep, B > 0: test jacobi_sweep_blocked. */
void check_sweep(int Nx, int Ny, int B)
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

    if (B > 0)
        jacobi_sweep_blocked(Nx, Ny, B, U, V);
    else
        jacobi_sweep(Nx, Ny, U, V);

    double err = 0.0;
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++) {
            double d = V[j * L + i] - U[j * L + i] - h * h;
            if (d < 0)
                d = -d;
            if (d > err)
                err = d;
        }

    if (B > 0)
        printf("  blocked (B = %d): ", B);
    else
        printf("  without blocking: ");
    if (err / (h * h) < 1e-6)
        printf("max |V - U - h^2| / h^2 = %.1e  OK\n", err / (h * h));
    else
        printf("max |V - U - h^2| / h^2 = %.1e  WRONG\n", err / (h * h));

    free(U);
    free(V);
}

/* Memory bandwidth in GB/s of the triad A[k] = B[k] + s*C[k] (lecture 4),
 * minimum time of 10 runs. Per k, 32 Byte are transferred: load B[k], C[k],
 * A[k] (write allocate) and store A[k]. */
double measure_bandwidth(int n)
{
    double *A = new_array(n);
    double *B = new_array(n);
    double *C = new_array(n);
    double s = 2.0;
    for (int k = 0; k < n; k++) {
        B[k] = 1.0;
        C[k] = 0.5;
    }

    double tmin = 1e30;
    for (int r = 0; r < 10; r++) {
        double t0 = wtime();
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
            A[k] = B[k] + s * C[k];
        double t = wtime() - t0;
        if (t < tmin)
            tmin = t;
    }
    if (A[n / 2] != 2.0)
        printf("Error in the triad\n");

    free(A);
    free(B);
    free(C);
    return 32.0 * n / tmin * 1e-9;
}

/* (b), (c): performance for one grid Nx x Ny, in million updates per second.
 * Minimal time of 5 sweeps; the versions with and without blocking are
 * measured alternately, so that both see the same state of the CPU
 * (clock frequency, temperature). */
void measure_performance(int Nx, int Ny, int B)
{
    double n = (double) Nx * Ny;   /* number of interior points */
    double *U = new_array((Nx + 2) * (Ny + 2));
    double *V = new_array((Nx + 2) * (Ny + 2));
    for (int k = 0; k < (Nx + 2) * (Ny + 2); k++)
        U[k] = 1.0;

    double tmin = 1e30, tmin_b = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = wtime();
        jacobi_sweep(Nx, Ny, U, V);
        double t = wtime() - t0;
        if (t < tmin)
            tmin = t;

        if (B > 0) {
            t0 = wtime();
            jacobi_sweep_blocked(Nx, Ny, B, U, V);
            t = wtime() - t0;
            if (t < tmin_b)
                tmin_b = t;
        }
    }

    printf("%8d %8d   %12.1f", Nx, Ny, n / tmin * 1e-6);
    if (B > 0)
        printf("   %12.1f", n / tmin_b * 1e-6);
    printf("\n");

    free(U);
    free(V);
}

int main(int argc, char **argv)
{
    int B = 0;   /* block width for jacobi_sweep_blocked, 0: no blocking */
    if (argc > 1)
        B = atoi(argv[1]);

    printf("Check with u(x,y) = x^2 + y^2, Nx = 50, Ny = 30:\n");
    check_sweep(50, 30, 0);
    if (B > 0) {
        check_sweep(50, 30, 7);
        check_sweep(50, 30, B);
    }

    printf("\nMemory bandwidth (triad): %.1f GB/s\n", measure_bandwidth(1 << 23));

    printf("\nOne Jacobi sweep, Nx * Ny = 2^24 (million updates per second):\n");
    printf("%8s %8s   %12s", "Nx", "Ny", "no blocking");
    if (B > 0)
        printf("   %12s", "blocked");
    printf("\n");
    for (int e = 8; e <= 21; e++) {
        int Nx = 1 << e;
        int Ny = (1 << 24) / Nx;
        measure_performance(Nx, Ny, B);
    }
    return 0;
}
