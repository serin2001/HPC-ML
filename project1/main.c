#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "jacobi.h"
#include "stencils.h"

/* ========================================================================= */
/* ORIGINAL-MODULE AUS DER VORLAGE (UNVERÄNDERT)                             */
/* ========================================================================= */

double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

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
            if (d < 0) d = -d;
            if (d > err) err = d;
        }

    if (B > 0)  printf("  blocked (B = %d): ", B);
    else        printf("  without blocking: ");
    
    if (err / (h * h) < 1e-6)   printf("max |V - U - h^2| / h^2 = %.1e  OK\n", err / (h * h));
    else                        printf("max |V - U - h^2| / h^2 = %.1e  WRONG\n", err / (h * h));

    free(U); free(V);
}

double measure_bandwidth(int n)
{
    double *A = new_array(n);
    double *B = new_array(n);
    double *C = new_array(n);
    double s = 2.0;
    for (int k = 0; k < n; k++) {
        B[k] = 1.0; C[k] = 0.5;
    }
    double tmin = 1e30;
    for (int r = 0; r < 10; r++) {
        double t0 = wtime();
        #pragma omp parallel for
        for (int k = 0; k < n; k++)
            A[k] = B[k] + s * C[k];
        double t = wtime() - t0;
        if (t < tmin) tmin = t;
    }
    if (A[n / 2] != 2.0) printf("Error in the triad\n");
    free(A); free(B); free(C);
    return 32.0 * n / tmin * 1e-9;
}

void measure_performance(int Nx, int Ny, int B)
{
    double n = (double) Nx * Ny;
    double *U = new_array((Nx + 2) * (Ny + 2));
    double *V = new_array((Nx + 2) * (Ny + 2));
    for (int k = 0; k < (Nx + 2) * (Ny + 2); k++) U[k] = 1.0;

    double tmin = 1e30, tmin_b = 1e30;
    for (int r = 0; r < 5; r++) {
        double t0 = wtime();
        jacobi_sweep(Nx, Ny, U, V);
        double t = wtime() - t0;
        if (t < tmin) tmin = t;

        if (B > 0) {
            t0 = wtime();
            jacobi_sweep_blocked(Nx, Ny, B, U, V);
            t = wtime() - t0;
            if (t < tmin_b) tmin_b = t;
        }
    }
    printf("%8d %8d   %12.1f", Nx, Ny, n / tmin * 1e-6);
    if (B > 0) printf("   %12.1f", n / tmin_b * 1e-6);
    printf("\n");
    free(U); free(V);
}

/* ========================================================================= */
/* ERWEITERUNGEN FÜR DIE NEUEN ANFORDERNUNGEN (k-Sweeps & Stencils)          */
/* ========================================================================= */

// Generischer Validierungstest für die neuen Stencils (CPU-seitig)
void check_sweep_generic(int Nx, int Ny, int B, const Stencil *S)
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

    if (B > 0) jacobi_sweep_blocked_generic(Nx, Ny, B, S, U, V);
    else       jacobi_sweep_generic(Nx, Ny, S, U, V);

    double err = 0.0;
    double expected_offset = S->test_factor * h * h;

    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            double d = V[j * L + i] - U[j * L + i] - expected_offset;
            if (d < 0) d = -d;
            if (d > err) err = d;
        }
    }

    printf("  %s ", S->name);
    if (B > 0) printf("(B = %d): ", B);
    else       printf("without blocking: ");
    
    if (err / (h * h) < 1e-6)   printf("OK\n");
    else                        printf("WRONG (err/h^2 = %.1e)\n", err / (h * h));

    free(U); free(V);
}



// Test of solve_jacobi with even and odd k (detects the old bug, where the
// result was freed for odd k): compare with k sweeps done by hand.
void check_solve(int Nx, int Ny, const Stencil *S, int k)
{
    int len = (Nx + 2) * (Ny + 2);
    double *U  = new_array(len);
    double *A  = new_array(len);
    double *Bv = new_array(len);

    srand(1);
    for (int q = 0; q < len; q++)
        U[q] = A[q] = Bv[q] = (double) rand() / RAND_MAX;   // incl. boundary

    JacobiResult r = solve_jacobi(Nx, Ny, S, U, k, 0.0, 0);

    double *cu = A, *cv = Bv;                 // reference: k sweeps by hand
    for (int s = 0; s < k; s++) {
        jacobi_sweep_generic(Nx, Ny, S, cu, cv);
        double *tmp = cu; cu = cv; cv = tmp;
    }

    double err = 0.0;
    for (int q = 0; q < len; q++) {
        double d = fabs(r.sol_ptr[q] - cu[q]);
        if (d > err) err = d;
    }
    printf("  solve_jacobi, %s, k = %d: %s\n", S->name, k,
           (r.iterations_completed == k && r.sol_ptr == U && err == 0.0) ? "OK" : "WRONG");
    free(U); free(A); free(Bv);
}

#ifdef USE_CUDA
// all GPU tests (the functions are in jacobi_gpu.cu)
void run_gpu_tests(void)
{
    const Stencil *stencils[3] = { &S5, &S9, &S5D };

    printf("\nGPU check with u(x,y) = x^2 + y^2:\n");
    for (int s = 0; s < 3; s++) {
        check_sweep_gpu(50, 30, stencils[s]);    // test of Exercise 2
        check_sweep_gpu(1, 1, stencils[s]);      // smallest grid
        check_sweep_gpu(33, 9, stencils[s]);     // not a multiple of 32 / 8
    }

    printf("\nGPU against CPU (random initial values):\n");
    for (int s = 0; s < 3; s++) {
        compare_cpu_gpu(50, 30, stencils[s], 1);
        compare_cpu_gpu(50, 30, stencils[s], 5);      // odd k
        compare_cpu_gpu(1000, 777, stencils[s], 10);  // even k, larger grid
    }
}
#endif

/* ========================================================================= */
/* ERWEITERTE TABELLEN-PERFORMANCE FÜR K-SWEEPS                              */
/* ========================================================================= */

// Performance table for one stencil: k sweeps per measurement, Nx*Ny = 2^24
void print_performance_table_cpu(const Stencil *S, int k_sweeps, int B)
{
    printf("\nPerformance for %s (%d sweeps, million updates per second):\n",
           S->name, k_sweeps);
    printf("%8s %8s   %12s", "Nx", "Ny", "no blocking");
    if (B > 0)
        printf("   %12s", "blocked");
    printf("\n");

    for (int e = 8; e <= 21; e++) {          // Nx = 2^8, ..., 2^21
        int Nx = 1 << e;
        int Ny = (1 << 24) / Nx;             // number of points stays 2^24

        double n = (double) Nx * Ny;
        double *U = new_array((Nx + 2) * (Ny + 2));
        double *V = new_array((Nx + 2) * (Ny + 2));
        for (int i = 0; i < (Nx + 2) * (Ny + 2); i++) { U[i] = 1.0; V[i] = 1.0; }

        double tmin = 1e30, tmin_b = 1e30;
      //  for (int r = 0; r < 5; r++) {
            double *cu = U, *cv = V;
            double t0 = wtime();
            for (int step = 0; step < k_sweeps; step++) {
                jacobi_sweep_generic(Nx, Ny, S, cu, cv);
                double *tmp = cu; cu = cv; cv = tmp;     // swap after EVERY sweep
            }
            double t = (wtime() - t0) / k_sweeps;
            if (t < tmin) tmin = t;

            if (B > 0) {
                cu = U; cv = V;
                t0 = wtime();
                for (int step = 0; step < k_sweeps; step++) {
                    jacobi_sweep_blocked_generic(Nx, Ny, B, S, cu, cv);
                    double *tmp = cu; cu = cv; cv = tmp;
                }
                t = (wtime() - t0) / k_sweeps;
                if (t < tmin_b) tmin_b = t;
            }
       // }

        printf("%8d %8d   %12.1f", Nx, Ny, n / tmin * 1e-6);
        if (B > 0)
            printf("   %12.1f", n / tmin_b * 1e-6);
        printf("\n");
        free(U); free(V);
    }
}
#ifdef USE_CUDA
void print_performance_table_gpu(const Stencil *S, int k_sweeps)
{
    printf("\nPerformance for %s on GPU (%d sweeps, million updates per second):\n",
           S->name, k_sweeps);
    printf("%8s %8s   %12s\n", "Nx", "Ny", "GPU Kernel");

    for (int e = 8; e <= 21; e++) {          // Nx = 2^8, ..., 2^21
        int Nx = 1 << e;
        int Ny = (1 << 24) / Nx;             // Gesamtzahl der Punkte bleibt 2^24

        double n = (double) Nx * Ny;
        size_t len = (size_t)(Nx + 2) * (Ny + 2);
        
        // Host-Speicher allozieren und initialisieren
        double *U = new_array(len);
        for (size_t i = 0; i < len; i++) { U[i] = 1.0; }

        // Aufruf deiner funktionierenden GPU-Methode!
        // Sie übernimmt das Allozieren, den Hin-Transfer, das k-fache Sweepen,
        // das Zeigerswapping auf dem Device und den Rücktransfer.
        GpuTimes times = solve_jacobi_gpu(Nx, Ny, S, U, k_sweeps);

        // Um exakt dasselbe zu messen wie bei der CPU (Zeit pro einzelnem Sweep):
        // teilen wir die gemessene reine Rechenzeit (t_sweeps) durch k_sweeps.
        double t_per_sweep = times.t_sweeps / k_sweeps;

        // Ausgabe im exakt selben Tabellenformat (Updates pro Sekunde)
        printf("%8d %8d   %12.1f\n", Nx, Ny, n / t_per_sweep * 1e-6);

        free(U);
    }
}
#endif
// // Hilfsfunktion nur für die finale Zusammenfassung (Single Run für 1024x1024)
// double get_single_run_mups_cpu(int Nx, int Ny, int B, const Stencil *S, int k)
// {
//     double n = (double) Nx * Ny;
//     double *U = new_array((Nx + 2) * (Ny + 2));
//     double *V = new_array((Nx + 2) * (Ny + 2));
//     for (int i = 0; i < (Nx + 2) * (Ny + 2); i++) U[i] = 1.0;

//     double tmin = 1e30;
//     for (int r = 0; r < 5; r++) {
//         double t0 = wtime();
//         double *current_U = U; double *current_V = V;
//         for (int step = 0; step < k; step++) {
//             if (B > 0) jacobi_sweep_blocked_generic(Nx, Ny, B, S, current_U, current_V);
//             else       jacobi_sweep_generic(Nx, Ny, S, current_U, current_V);
//             double *tmp = current_U; current_U = current_V; current_V = tmp;
//         }
//         double t = wtime() - t0;
//         if (t < tmin) tmin = t;
//     }
//     free(U); free(V);
//     return (n * k) / tmin * 1e-6;
// }

/* ========================================================================= */
/* MAIN-STEUERUNG                                                            */
/* ========================================================================= */
int main(int argc, char **argv)
{
    int B = (argc > 1) ? atoi(argv[1]) : 0;   // block width, 0: no blocking
    int k_sweeps = 50;          // sweeps per measurement in the tables
    int maxIter = 1000;
    int check_every = 100;        // residual only every 10 sweeps (costs a sweep)
    int Nx = 256; //2048;
    int Ny = 256; //2048;
    double epsilon = 1e-4;
    int len = (Nx + 2) * (Ny + 2);
    double *U = new_array(len);
    double *U_master = new_array(len);

    // initial values: random in the interior, boundary = 0
    srand(42);
    for (int j = 0; j < Ny + 2; j++)
        for (int i = 0; i < Nx + 2; i++) {
            int idx = j * (Nx + 2) + i;
            if (i == 0 || i == Nx + 1 || j == 0 || j == Ny + 1)
                U_master[idx] = 0.0;
            else
                U_master[idx] = (double) rand() / (double) RAND_MAX;
        }

    // 1. tests (with "crooked" grid sizes and with/without blocking)
    printf("Check with u(x,y) = x^2 + y^2, Nx = 50, Ny = 30:\n");
    const Stencil *stencils[3] = { &S5, &S9, &S5D };
    for (int s = 0; s < 3; s++) {
        check_sweep_generic(50, 30, 0, stencils[s]);
        check_sweep_generic(50, 30, 7, stencils[s]);
        if (B > 0)
            check_sweep_generic(50, 30, B, stencils[s]);
    }
    printf("Check of solve_jacobi (even and odd number of sweeps):\n");
    for (int s = 0; s < 3; s++) {
        check_solve(50, 30, stencils[s], 4);
        check_solve(50, 30, stencils[s], 5);
    }

    // 2. memory bandwidth
    printf("\nMemory bandwidth (triad): %.1f GB/s\n", measure_bandwidth(1 << 23));

    // 3. performance tables (uncomment if needed, they take a while)
    for (int s = 0; s < 3; s++)
    //    print_performance_table_cpu(stencils[s], k_sweeps, B);
    (void) k_sweeps;

    // 4. Jacobi solver for all three stencils, same initial values
    for (int s = 0; s < 3; s++) {
        memcpy(U, U_master, len * sizeof(double));
        JacobiResult result = solve_jacobi(Nx, Ny, stencils[s], U, maxIter,
                                           epsilon, check_every);
        printf("\nStencil: %s\n", stencils[s]->name);
        printf("Grid: %d x %d\n", Nx, Ny);
        printf("Time for %d sweeps: %f s (%.1f million updates/s)\n",
               result.iterations_completed, result.elapsed_time,
               (double) Nx * Ny * result.iterations_completed
                   / result.elapsed_time * 1e-6);
        printf("Residual norm (L2, every %d sweeps): %f\n",
               check_every, result.final_residual);
    }

#ifdef USE_CUDA
    printf("\n--- GPU PATH ACTIVE (CLUSTER MODE) ---\n");
    run_gpu_tests();
	
    double bw_gpu = measure_bandwidth_gpu(1 << 23);
    printf("\n=== STREAM TRIAD BANDWIDTH COMPARISON ===\n");
    //printf("CPU Bandwidth: %.2f GB/s\n", bw_cpu);
    printf("GPU Bandwidth (triad): %.2f GB/s\n", bw_gpu);
    
        for (int s = 0; s < 3; s++) {
        printf("\n======================================================\n");
        printf("Stencil: %s | Grid: %d x %d\n", stencils[s]->name, Nx, Ny);
        printf("======================================================\n");

        // ------------------------------------------------------------
        // BERECHNUNG AUF DER GPU (Deine Portierung!)
        // ------------------------------------------------------------
        
    

    
        // Wir setzen das Arbeits-Array U wieder exakt auf die CPU-Startmatrix zurück
        memcpy(U, U_master, len * sizeof(double)); 
        
        // Aufruf deiner portierten GPU-Methode
print_performance_table_gpu(stencils[s], k_sweeps);
    }
#endif

    free(U);
    free(U_master);
    return 0;
}
