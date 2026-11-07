#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "jacobi.h"

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



/* ========================================================================= */
/* ERWEITERTE TABELLEN-PERFORMANCE FÜR K-SWEEPS                              */
/* ========================================================================= */

// Hilfsfunktion, die eine vollständige Tabelle für ein Stencil generiert
void print_performance_table_cpu(const Stencil *S, int k_sweeps, int B)
{
    printf("\nPerformance for %s (%d sweeps):\n", S->name, k_sweeps);
    //printf("\nOne Jacobi sweep, Nx * Ny = 2^24 (million updates per second):\n");
    printf("%8s %8s   %12s", "Nx", "Ny", "no blocking");
    if (B > 0)
        printf("   %12s", "blocked");
    printf("\n");
    
    for (int e = 8; e <= 21; e++) { // Läuft von Nx=256 bis Nx=65536
        int Nx = 1 << e;
        int Ny = (1 << 24) / Nx; // Gesamtzahl Punkte bleibt 2^24
        
        double n = (double) Nx * Ny;
        double *U = new_array((Nx + 2) * (Ny + 2));
        double *V = new_array((Nx + 2) * (Ny + 2));
        for (int i = 0; i < (Nx + 2) * (Ny + 2); i++) U[i] = 1.0;

        double tmin = 1e30, tmin_b = 1e30;
        for (int r = 0; r < 5; r++) {
            double t0 = wtime();            
            double *current_U = U;
            double *current_V = V;
            for (int step = 0; step < k_sweeps; step++) 
                jacobi_sweep_generic(Nx, Ny, S, current_U, current_V);
        
            double t = (wtime() - t0) / k_sweeps;
            if (t < tmin)
                tmin = t;

            if (B > 0) {
                t0 = wtime();
                for (int step = 0; step < k_sweeps; step++) 
                    jacobi_sweep_blocked(Nx, Ny, B, U, V);
                t = (wtime() - t0) / k_sweeps;
                if (t < tmin_b)
                    tmin_b = t;
            }
            double *tmp = current_U; current_U = current_V; current_V = tmp;
        }

        printf("%8d %8d   %12.1f", Nx, Ny, n / tmin * 1e-6);
        if (B > 0)
            printf("   %12.1f", n / tmin_b * 1e-6);
        printf("\n");       
        free(U); free(V);
    }
}

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
    int B = (argc > 1) ? atoi(argv[1]) : 0;
    int k_sweeps = 3; //100; 
    int maxIter = 100;
    int Nx = 2048; // 512; // 128; // 32;
    int Ny = 2048; // 512; // 128; // 32;
    double epsilon = 1e-4;
    double *U = new_array((Nx + 2) * (Ny + 2));
    double *U_master = new_array((Nx + 2) * (Ny + 2));
    //for (int i = 0; i < (Nx + 2) * (Ny + 2); i++) U[i] = 1.0;
    //for (int i = 0; i < Nx + 2; i++) {
    //U[i] = 0.0;          // Unterer Rand (j = 0)
    //U[(Ny + 1) * (Nx + 2) + i] = 0.0;   // Oberer Rand (j = Ny + 1)

// for (int j = 0; j < Ny + 2; j++) {
//     U[j * (Nx + 2) ] = 0.0;          // Linker Rand (i = 0)
//     U[j * (Nx + 2) + (Nx + 1)] = 0.0;   // Rechter Rand (i = Nx + 1)
//}
srand(42);
for (int j = 0; j < Ny + 2; j++) {
    for (int i = 0; i < Nx + 2; i++) {
        int idx = j * (Nx + 2) + i;
        
        // Prüfen, ob wir uns am Rand befinden
        if (i == 0 || i == Nx + 1 || j == 0 || j == Ny + 1) {
            U_master[idx] = 0.0; // Randbedingung
        } else {
            // Zufallswert zwischen 0.0 und 1.0
            U_master[idx] = (double)rand() / (double)RAND_MAX; 
        }
    }
}

    printf("Check with u(x,y) = x^2 + y^2, Nx = 50, Ny = 30:\n");
    check_sweep_generic(50, 30, B, &S5);
    check_sweep_generic(50, 30, B, &S9);
    check_sweep_generic(50, 30, B, &S5D);

    // 3. Originale Bandbreitenmessung
    printf("\nMemory bandwidth (triad): %.1f GB/s\n", measure_bandwidth(1 << 23));
    
    // 4. Tabellen-Ausgaben für alle 3 Stencils über 100 Sweeps
    // printf("\n--- PERFORMANCE TABLES (%d SWEEPS, CPU) ---\n", 1);
    // print_performance_table_cpu(&S5, 1, B);
    // print_performance_table_cpu(&S9, 1, B);
    // print_performance_table_cpu(&S5D, 1, B);

    // 4. Tabellen-Ausgaben für alle 3 Stencils über 100 Sweeps
    // printf("\n--- PERFORMANCE TABLES (%d SWEEPS, CPU) ---\n", k_sweeps);
    // print_performance_table_cpu(&S5, k_sweeps, B);
    // print_performance_table_cpu(&S9, k_sweeps, B);
    // print_performance_table_cpu(&S5D, k_sweeps, B);

    JacobiResult result;

    memcpy(U, U_master, (Nx + 2) * (Ny + 2) * sizeof(double));
result = solve_jacobi(Nx, Ny, &S9, U, maxIter, epsilon);

printf("Stencil: %s\n", S9.name);
printf("Gitter: %d x %d\n", Nx, Ny);
printf("Dauer für %d Sweeps: %f Sekunden\n", result.iterations_completed, result.elapsed_time);
printf("Residualnorm (L2): %f\n", result.final_residual);

    memcpy(U, U_master, (Nx + 2) * (Ny + 2) * sizeof(double));
    result = solve_jacobi(Nx, Ny, &S5, U, maxIter, epsilon);

printf("Stencil: %s\n", S5.name);
printf("Gitter: %d x %d\n", Nx, Ny);
printf("Dauer für %d Sweeps: %f Sekunden\n", result.iterations_completed, result.elapsed_time);
printf("Residualnorm (L2): %f\n", result.final_residual);

memcpy(U, U_master, (Nx + 2) * (Ny + 2) * sizeof(double));
result = solve_jacobi(Nx, Ny, &S5D, U, maxIter, epsilon);

printf("Stencil: %s\n", S5D.name);
printf("Gitter: %d x %d\n", Nx, Ny);
printf("Dauer für %d Sweeps: %f Sekunden\n", result.iterations_completed, result.elapsed_time);
printf("Residualnorm (L2): %f\n", result.final_residual);

// // 5. Kompakte Zusammenfassung am Ende
    // printf("\n--- SUMMARY (Nx=1024, Ny=1024, k=%d) ---\n", k_sweeps);
    // printf("CPU S5 : %.1f Mups/s\n", get_single_run_mups_cpu(1024, 1024, B, &S5, k_sweeps));
    // printf("CPU S9 : %.1f Mups/s\n", get_single_run_mups_cpu(1024, 1024, B, &S9, k_sweeps));
    // printf("CPU S5D: %.1f Mups/s\n", get_single_run_mups_cpu(1024, 1024, B, &S5D, k_sweeps));

#ifdef USE_CUDA
    printf("\n--- GPU PATH ACTIVE (CLUSTER MODE) ---\n");
    // Hier drucken wir später die analogen GPU-Tabellen zum direkten Vergleich!
#endif

    return 0;
}
