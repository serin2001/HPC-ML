#ifndef STENCILS_H
#define STENCILS_H

/*
 * A stencil S (multiplied by h^2) approximates h^2 * (-Laplace).
 * Orientation as in Exercise 1 (the stencil looks like it does on the grid):
 *     matrix[0][.] = row j+1 (top),  matrix[1][.] = row j,  matrix[2][.] = row j-1 (bottom)
 *     matrix[.][0] = column i-1,     matrix[.][1] = column i, matrix[.][2] = column i+1
 *
 * One sweep:  V = U - (1/sc) * S U,  sc = matrix[1][1].
 * Test with u = x^2 + y^2:  S U = -4 h^2 exactly, so V = U + (4/sc) h^2,
 * i.e. test_factor = 4 / sc.
 */
typedef struct {
    const char* name;
    double matrix[3][3];
    double sc;          // centre coefficient, must equal matrix[1][1]
    double test_factor; // 4.0 / sc  (factor for the test with u = x^2 + y^2)
} Stencil;

typedef struct {
    int iterations_completed; // number of sweeps actually done
    double elapsed_time;      // measured time in seconds
    double final_residual;    // last computed residual norm
    double *sol_ptr;          // result; always the array U_start passed in
} JacobiResult;

/* The stencil definitions use C designated initializers; they are only
 * needed in the C files (the .cu file only needs the types above). */
#ifndef __cplusplus

static const Stencil S5 = {
    .name = "5-Point Stencil (S5)",
    .matrix = {
        { 0.0, -1.0,  0.0},
        {-1.0,  4.0, -1.0},
        { 0.0, -1.0,  0.0}
    },
    .sc = 4.0,
    .test_factor = 1.0          // 4 / 4
};

static const Stencil S9 = {
    .name = "Compact 9-Point Stencil (S9)",
    .matrix = {
        {-1.0/6.0, -4.0/6.0, -1.0/6.0},
        {-4.0/6.0, 20.0/6.0, -4.0/6.0},
        {-1.0/6.0, -4.0/6.0, -1.0/6.0}
    },
    .sc = 20.0 / 6.0,
    .test_factor = 1.2          // 4 / (20/6)
};

/* Diagonal ("rotated") five-point stencil. The neighbours are at distance
 * sqrt(2) h, so the stencil with weights 4,-1,-1,-1,-1 approximates
 * 2 h^2 (-Laplace); divided by 2 it approximates h^2 (-Laplace) like the
 * others, and test_factor = 4/sc holds. (The sweep itself is the same.) */
static const Stencil S5D = {
    .name = "Diagonal 5-Point Stencil (S5D)",
    .matrix = {
        {-0.5,  0.0, -0.5},
        { 0.0,  2.0,  0.0},
        {-0.5,  0.0, -0.5}
    },
    .sc = 2.0,
    .test_factor = 2.0          // 4 / 2
};

#endif /* __cplusplus */

#endif
