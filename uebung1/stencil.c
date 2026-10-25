/*
 * Exercise 1, programming task: matrix-free application of the five-point
 * and the nine-point stencil for -Laplace(u)
 * HPC for ML, WS 2026/27
 *
 * Compile: make        (builds stencil, stencil_O0 and stencil_omp)
 * Run:     ./stencil
 *          OMP_NUM_THREADS=4 ./stencil_omp
 *
 * Grid: N interior points per direction, h = 1/(N+1).
 * A grid function is stored INCLUDING the boundary points in an array of
 * length (N+2)*(N+2), row by row:  u_{i,j} = U[j*(N+2) + i],  i,j = 0,...,N+1.
 * The boundary values are zero.
 *
 * A stencil S[3][3] is stored as it looks on the grid (multiplied by h^2):
 * S[0][.] is the row j+1, S[1][.] the row j, S[2][.] the row j-1, and
 * S[.][0] is the column i-1, S[.][1] the column i, S[.][2] the column i+1.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/* ------------------------------------------------------------------------ */
/* (a) matrix-free application of a stencil: Y = A U                        */
/* ------------------------------------------------------------------------ */

/* For all interior points (i,j):
 *     Y_{i,j} = sum over di,dj = -1,0,1 of  S[1-dj][1+di] * U_{i+di,j+dj}
 * The boundary values of Y are not changed. */
void apply_stencil(int N, double S[3][3], double *U, double *Y)
{
    /* TODO (a); in part (c), parallelize it with OpenMP */
    #pragma omp parallel for
    for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= N; i++) {

            double sum = 0.0;

            for (int dj = -1; dj <= 1; dj++) {
                for (int di = -1; di <= 1; di++) {
                    sum += S[1 - dj][1 + di] * U[(j + dj) * (N+2) + (i + di)];
                }
            }

            Y[j * (N+2) + i] = sum;
        }
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

/* array for a grid function, all values (including the boundary) set to 0 */
double *new_grid(int N)
{
    int len = (N + 2) * (N + 2);
    double *U = (double *) malloc(len * sizeof(double));
    if (U == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (int k = 0; k < len; k++)
        U[k] = 0.0;
    return U;
}

/* Test with u(x,y) = x(1-x) y(1-y), which is zero on the boundary, and
 * -Laplace(u) = f(x,y) = 2 ( x(1-x) + y(1-y) ).
 * Prints max |Y/h^2 - f| / max |f|, which must be small. */
void check_stencil(int N, double S[3][3], char *name)
{
    double h = 1.0 / (N + 1);
    double *U = new_grid(N);
    double *Y = new_grid(N);

    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= N; i++) {
            double x = i * h, y = j * h;
            U[j * (N + 2) + i] = x * (1.0 - x) * y * (1.0 - y);
        }

    apply_stencil(N, S, U, Y);

    double err = 0.0, nrm = 0.0;
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= N; i++) {
            double x = i * h, y = j * h;
            double f = 2.0 * (x * (1.0 - x) + y * (1.0 - y));
            double d = Y[j * (N + 2) + i] / (h * h) - f;
            if (d < 0)
                d = -d;
            if (d > err)
                err = d;
            if (f > nrm)
                nrm = f;
        }

    if (err / nrm < 1e-3)
        printf("  %s: max |Y/h^2 - f| / max |f| = %.1e  OK\n", name, err / nrm);
    else
        printf("  %s: max |Y/h^2 - f| / max |f| = %.1e  WRONG\n", name, err / nrm);

    free(U);
    free(Y);
}

/* minimal runtime of reps calls of apply_stencil */
double time_stencil(int N, double S[3][3], double *U, double *Y, int reps)
{
    double tmin = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = wtime();
        apply_stencil(N, S, U, Y);
        double t = wtime() - t0;
        if (t < tmin)
            tmin = t;
    }
    return tmin;
}

/* (b): runtime of both stencils for one grid size N (minimum of 10 runs) */
void measure_performance(int N, double S5[3][3], double S9[3][3])
{
    int n = N * N;   /* number of interior points */
    double *U = new_grid(N);
    double *Y = new_grid(N);
    for (int j = 1; j <= N; j++)     /* boundary values stay 0 */
        for (int i = 1; i <= N; i++)
            U[j * (N + 2) + i] = 1.0;

    double t5 = time_stencil(N, S5, U, Y, 10);
    double t9 = time_stencil(N, S9, U, Y, 10);
    printf("%6d %10d   %10.3e %10.1f   %10.3e %10.1f\n",
           N, n, t5, n / t5 * 1e-6, t9, n / t9 * 1e-6);

    free(U);
    free(Y);
}

int main(void)
{
    /* five-point stencil from the lecture, multiplied by h^2 */
    double S5[3][3] = {
        { 0.0, -1.0,  0.0},
        {-1.0,  4.0, -1.0},
        { 0.0, -1.0,  0.0}
    };

    /* (a) nine-point stencil from Task 1, multiplied by h^2 */
    double S9[3][3] = {
        {-1.0/6, -4.0/6, -1.0/6},   /* TODO */
        {-4.0/6, 20.0/6, -4.0/6},
        {-1.0/6, -4.0/6, -1.0/6}
    };

    printf("Check with u(x,y) = x(1-x) y(1-y), N = 63:\n");
    check_stencil(63, S5, "5-point");
    check_stencil(63, S9, "9-point");

    printf("\nRuntime of apply_stencil (time in s, million updates per second):\n");
    printf("%6s %10s   %21s   %21s\n", "N", "n = N*N", "5-point", "9-point");
    measure_performance(255, S5, S9);
    measure_performance(511, S5, S9);
    measure_performance(1023, S5, S9);
    measure_performance(2047, S5, S9);
    return 0;
}
