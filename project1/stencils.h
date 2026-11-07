#ifndef STENCILS_H
#define STENCILS_H

typedef struct {
    const char* name;
    double matrix[3][3];
    double sc;          // Zentrumskoeffizient
    double test_factor; // Der Faktor für den u = x^2 + y^2 Test (4.0 / sc)
} Stencil;

typedef struct {
    int iterations_completed; // Wie viele Sweeps wurden tatsächlich gemacht?
    double elapsed_time;       // Gemessene Zeit auf CPU (oder GPU) in Sekunden
    double final_residual;     // Die letzte berechnete Fehlernorm
    double *sol_ptr;           // Zeiger auf das finale Lösungsarray (entweder U oder V)
} JacobiResult;

// Definition der drei verlangten Stencils
static const Stencil S5 = {
    .name = "5-Point Stencil (S5)",
    .matrix = {
        { 0.0, -1.0,  0.0},
        {-1.0,  4.0, -1.0},
        { 0.0, -1.0,  0.0}
    },
    .sc = 4.0,
    .test_factor = 1.0 // 4.0 / 4.0
};

static const Stencil S9 = {
    .name = "Kompakt 9-Point Stencil (S9)",
    .matrix = {
        {-1.0/6.0, -4.0/6.0, -1.0/6.0},
        {-4.0/6.0, 20.0/6.0, -4.0/6.0},
        {-1.0/6.0, -4.0/6.0, -1.0/6.0}
    },
    .sc = 20.0 / 6.0,
    .test_factor = 1.2 // 4.0 / (20.0/6.0) = 24/20 = 1.2
};

static const Stencil S5D = {
    .name = "Diagonal 5-Point Stencil (S5D)",
    .matrix = {
        {-1.0,  0.0, -1.0},
        { 0.0,  4.0,  0.0},
        {-1.0,  0.0, -1.0}
    },
    .sc = 4.0,
    .test_factor = 2.0 // 8.0 / 4.0
};

#endif
