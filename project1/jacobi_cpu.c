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
#include <string.h>
#include <math.h>
#include "stencils.h"
#include "jacobi.h"

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

void jacobi_sweep_generic(int Nx, int Ny, const Stencil *S, const double *U, double *V)
{
    int L = Nx + 2;
    double inv_sc = 1.0 / S->sc;

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            // Explizites Ausmultiplizieren der 3x3-Nachbarschaft (Vektorisierungs-freundlich)
            double SU = 
                S->matrix[0][0] * U[(j - 1) * L + i - 1] + S->matrix[0][1] * U[(j - 1) * L + i] + S->matrix[0][2] * U[(j - 1) * L + i + 1] +
                S->matrix[1][0] * U[    j     * L + i - 1] + S->matrix[1][1] * U[    j     * L + i] + S->matrix[1][2] * U[    j     * L + i + 1] +
                S->matrix[2][0] * U[(j + 1) * L + i - 1] + S->matrix[2][1] * U[(j + 1) * L + i] + S->matrix[2][2] * U[(j + 1) * L + i + 1];

            V[j * L + i] = U[j * L + i] - (SU * inv_sc);
        }
    }
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
void jacobi_sweep_blocked_generic(int Nx, int Ny, int B, const Stencil *S, const double *U, double *V)
{
    int L = Nx + 2;
    double inv_sc = 1.0 / S->sc;

    for (int i_block = 1; i_block <= Nx; i_block += B) {
        int i_end = (i_block + B - 1 < Nx) ? i_block + B - 1 : Nx;

        #pragma omp parallel for schedule(static)
        for (int j = 1; j <= Ny; j++) {
            for (int i = i_block; i <= i_end; i++) {
                double SU = 
                    S->matrix[0][0] * U[(j - 1) * L + i - 1] + S->matrix[0][1] * U[(j - 1) * L + i] + S->matrix[0][2] * U[(j - 1) * L + i + 1] +
                    S->matrix[1][0] * U[    j     * L + i - 1] + S->matrix[1][1] * U[    j     * L + i] + S->matrix[1][2] * U[    j     * L + i + 1] +
                    S->matrix[2][0] * U[(j + 1) * L + i - 1] + S->matrix[2][1] * U[(j + 1) * L + i] + S->matrix[2][2] * U[(j + 1) * L + i + 1];

                V[j * L + i] = U[j * L + i] - (SU * inv_sc);
            }
        }
    }
}

// Berechnet die L1-Norm (Summe der Beträge) des Residuums
double compute_norm_l1(int Nx, int Ny, const Stencil *S, const double *U) {
    int L = Nx + 2;
    double total_sum = 0.0;

    // OpenMP parallelisiert die Schleife und führt die Summen-Reduktion sicher aus
    #pragma omp parallel for reduction(+:total_sum) schedule(static)
    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            // 1. Lokales Residuum r = A*U (da f = 0)
            double SU = 
                S->matrix[0][0] * U[(j - 1) * L + i - 1] + S->matrix[0][1] * U[(j - 1) * L + i] + S->matrix[0][2] * U[(j - 1) * L + i + 1] +
                S->matrix[1][0] * U[    j     * L + i - 1] + S->matrix[1][1] * U[    j     * L + i] + S->matrix[1][2] * U[    j     * L + i + 1] +
                S->matrix[2][0] * U[(j + 1) * L + i - 1] + S->matrix[2][1] * U[(j + 1) * L + i] + S->matrix[2][2] * U[(j + 1) * L + i + 1];

            // 2. Absolutbetrag aufaddieren
            total_sum += fabs(SU);
        }
    }
    return total_sum;
}


JacobiResult solve_jacobi(int Nx, int Ny, const Stencil *S, double *U_start, int kmax, double epsilon) {
    int L = Nx + 2;
    int total_size = L * (Ny + 2);
    
    // Wir brauchen ein zweites Array V für das Ping-Pong-Verfahren
    double n = (double) Nx * Ny;
    double res_norm = 10000000.0;
    double center_D = S->matrix[1][1];

    double *V = new_array((Nx + 2) * (Ny + 2));

    memcpy(V, U_start, total_size * sizeof(double));

    // Lokale Pointer für das fehlerfreie Ping-Pong-Verfahren
    double *U = U_start;
    double *V_next = V;
    double t0 = wtime();  
    int k = 0;
    for (k = 0; k < kmax; k++) {
        // Führe den Sweep aus
        jacobi_sweep_generic(Nx, Ny, S, U, V_next);

        double local_sum = 0.0;
            #pragma omp parallel for reduction(+:local_sum) schedule(static)
            for (int j = 1; j <= Ny; j++) {
                for (int i = 1; i <= Nx; i++) {
                    int idx = j * L + i;
                    // Residuum r = D * (U_alt - V_neu)
                    double r_local = center_D * (U[idx] - V_next[idx]);
                    local_sum += r_local * r_local;
                }
            }
            res_norm = sqrt(local_sum);
            if (k%10 == 9) printf("Residuum nach %d Iterationen: %f \n", k, res_norm);

        // Zeigertausch (Ping-Pong) -> Kostet O(1) Zeit
        double *tmp = U;
        U = V_next;
        V_next = tmp;
        
        if (res_norm < epsilon) {
            k++; // Den aktuell beendeten Sweep mitzählen
            break;
        }
    }
    double t = (wtime() - t0);
    // Ergebnis strukturieren
    JacobiResult res;
    res.iterations_completed = k;
    res.elapsed_time = t; //end_time - start_time;
    res.final_residual = res_norm; // Berechnen wir im nächsten Schritt
    res.sol_ptr = U;      // Zeigt auf das aktuellste Ergebnis

    // Wichtig: V_next zeigt jetzt auf das "alte" oder ungenutzte Array, 
    // aber wir müssen aufpassen, dass wir den originalen malloc-Pointer freigeben!
    // Am besten bufferst du den originalen V_buffer Pointer, um ihn hier zu free'n.
    if (U == U_start) {
        free(V_next); // Wenn U_curr wieder vorn steht, ist V_next das temporäre Array
    } else {
        free(U); // Andernfalls ist U_curr das temporäre Array
    }
    return res;
}
