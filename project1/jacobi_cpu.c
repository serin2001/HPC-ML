/*
 * Project 1, CPU part: Jacobi sweeps for the five-point stencil and for
 * general 3x3 stencils, spatial blocking, and the Jacobi solver with k sweeps.
 * HPC for ML, WS 2026/27
 *
 * Grid: Nx interior points in x-direction (index i), Ny interior points in
 * y-direction (index j). A grid function is stored INCLUDING the boundary
 * points in an array of length (Nx+2)*(Ny+2), row by row:
 *     u_{i,j} = U[j*(Nx+2) + i],   i = 0,...,Nx+1,  j = 0,...,Ny+1.
 *
 * General sweep for a stencil S (multiplied by h^2) with centre coefficient sc:
 *     V_{i,j} = U_{i,j} - (1/sc) (S U)_{i,j}
 * Stencil orientation as in Exercise 1 (see stencils.h):
 *     matrix[0][.] = row j+1, matrix[1][.] = row j, matrix[2][.] = row j-1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "stencils.h"
#include "jacobi.h"

/* The nine stencil weights are copied into local variables before the loops.
 * Otherwise the compiler has to assume that writing V may change S->matrix
 * (aliasing) and reloads all nine weights for every grid point. */
#define LOAD_STENCIL(S)                                                       \
    const double s00 = (S)->matrix[0][0], s01 = (S)->matrix[0][1],            \
                 s02 = (S)->matrix[0][2], s10 = (S)->matrix[1][0],            \
                 s11 = (S)->matrix[1][1], s12 = (S)->matrix[1][2],            \
                 s20 = (S)->matrix[2][0], s21 = (S)->matrix[2][1],            \
                 s22 = (S)->matrix[2][2]

/* (S U)_{i,j}; row j+1 belongs to matrix[0], row j-1 to matrix[2] */
#define APPLY_STENCIL(U, idx, L)                                              \
    (  s00 * U[(idx) + (L) - 1] + s01 * U[(idx) + (L)] + s02 * U[(idx) + (L) + 1] \
     + s10 * U[(idx)       - 1] + s11 * U[(idx)      ] + s12 * U[(idx)       + 1] \
     + s20 * U[(idx) - (L) - 1] + s21 * U[(idx) - (L)] + s22 * U[(idx) - (L) + 1])

/* ------------------------------------------------------------------------ */
/* Five-point Jacobi sweep (Exercise 2)                                     */
/* ------------------------------------------------------------------------ */

/* The boundary values of V are not changed. */
void jacobi_sweep(int Nx, int Ny, double *U, double *V)
{
    int L = Nx + 2;
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++)
            V[j * L + i] = 0.25 * (U[(j - 1) * L + i] + U[j * L + i - 1]
                                 + U[j * L + i + 1] + U[(j + 1) * L + i]);
}

void jacobi_sweep_blocked(int Nx, int Ny, int B, double *U, double *V)
{
    int L = Nx + 2;
    for (int i0 = 1; i0 <= Nx; i0 += B) {
        int i1 = (i0 + B - 1 < Nx) ? i0 + B - 1 : Nx;   /* last block shorter */
        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= Ny; j++)
            for (int i = i0; i <= i1; i++)
                V[j * L + i] = 0.25 * (U[(j - 1) * L + i] + U[j * L + i - 1]
                                     + U[j * L + i + 1] + U[(j + 1) * L + i]);
    }
}

/* ------------------------------------------------------------------------ */
/* General 3x3 stencil                                                      */
/* ------------------------------------------------------------------------ */

void jacobi_sweep_generic(int Nx, int Ny, const Stencil *S, const double *U, double *V)
{
    int L = Nx + 2;
    LOAD_STENCIL(S);
    const double inv_sc = 1.0 / s11;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            int idx = j * L + i;
            V[idx] = U[idx] - inv_sc * APPLY_STENCIL(U, idx, L);
        }
    }
}

void jacobi_sweep_blocked_generic(int Nx, int Ny, int B, const Stencil *S,
                                  const double *U, double *V)
{
    int L = Nx + 2;
    LOAD_STENCIL(S);
    const double inv_sc = 1.0 / s11;

    for (int i0 = 1; i0 <= Nx; i0 += B) {
        int i1 = (i0 + B - 1 < Nx) ? i0 + B - 1 : Nx;
        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= Ny; j++) {
            for (int i = i0; i <= i1; i++) {
                int idx = j * L + i;
                V[idx] = U[idx] - inv_sc * APPLY_STENCIL(U, idx, L);
            }
        }
    }
}

/* L1 norm of the residual  r = S U  (right-hand side f = 0) */
double compute_norm_l1(int Nx, int Ny, const Stencil *S, const double *U)
{
    int L = Nx + 2;
    LOAD_STENCIL(S);
    double total_sum = 0.0;

    #pragma omp parallel for reduction(+:total_sum) schedule(static)
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++) {
            int idx = j * L + i;
            total_sum += fabs(APPLY_STENCIL(U, idx, L));
        }
    return total_sum;
}

/* ------------------------------------------------------------------------ */
/* Jacobi solver: up to kmax sweeps with pointer swap (ping-pong)           */
/* ------------------------------------------------------------------------ */

/* U_start: initial values incl. boundary; on return it contains the result
 *          (also for an odd number of sweeps).
 * check_every: the residual is computed only every check_every sweeps
 *          (0: never, then exactly kmax sweeps are done). Computing it reads
 *          U and V once more, which costs about as much as a whole sweep,
 *          so it must not be done after every sweep when measuring times.
 * The residual is r = sc (U_old - V_new) = S U_old, i.e. the residual of the
 * iterate BEFORE the last sweep (it comes for free from the sweep). */
JacobiResult solve_jacobi(int Nx, int Ny, const Stencil *S, double *U_start,
                          int kmax, double epsilon, int check_every)
{
    int L = Nx + 2;
    size_t total_size = (size_t) L * (Ny + 2);
    double sc = S->matrix[1][1];
    double res_norm = -1.0;          /* -1: not computed */

    /* second array; it must have the same boundary values as U */
    double *V_buf = new_array((int) total_size);
    memcpy(V_buf, U_start, total_size * sizeof(double));

    double *U = U_start;
    double *V = V_buf;

    double t0 = wtime();
    int k;
    for (k = 0; k < kmax; k++) {
        jacobi_sweep_generic(Nx, Ny, S, U, V);

        int check = (check_every > 0) && ((k + 1) % check_every == 0);
        if (check) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static)
            for (int j = 1; j <= Ny; j++)
                for (int i = 1; i <= Nx; i++) {
                    int idx = j * L + i;
                    double r = sc * (U[idx] - V[idx]);
                    sum += r * r;
                }
            res_norm = sqrt(sum);
        }

        /* pointer swap: U points to the newest iterate */
        double *tmp = U;
        U = V;
        V = tmp;

        if (check && res_norm < epsilon) {
            k++;                      /* count the sweep just finished */
            break;
        }
    }
    double t = wtime() - t0;

    /* After an odd number of sweeps the result is in V_buf: copy it back,
     * so that the result is always in U_start. V_buf can then always be
     * freed (before, the result itself was freed in this case). */
    if (U != U_start)
        memcpy(U_start, U, total_size * sizeof(double));
    free(V_buf);

    JacobiResult res;
    res.iterations_completed = k;
    res.elapsed_time = t;            /* without the final copy */
    res.final_residual = res_norm;
    res.sol_ptr = U_start;
    return res;
}
