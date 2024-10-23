#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <stdlib.h> // For: exit, random, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>

//
// C-based micorkernel
//
#include <arm_sve.h>

void bl_dgemm_ukr(int kc,
                  int m,
                  int n,
                  double *A,
                  double *B,
                  double *C,
                  unsigned long long ldc,
                  aux_t* data)
{
    //int l, j, i;
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x,c2x,c3x,c4x,c5x,c6x,c7x,c8x,c9x,c10x,c11x,c12x,c13x,c14x,c15x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    c0x = svld1_f64(npred, C);
    c1x = (m > 1) ? svld1_f64(npred, C+ldc) : svdup_f64(0.0);
    c2x = (m > 2) ? svld1_f64(npred, C+2*ldc) : svdup_f64(0.0);
    c3x = (m > 3) ? svld1_f64(npred, C+3*ldc) : svdup_f64(0.0);
    c4x = (m > 4) ? svld1_f64(npred, C+4*ldc) : svdup_f64(0.0);
    c5x = (m > 5) ? svld1_f64(npred, C+5*ldc) : svdup_f64(0.0);
    c6x = (m > 6) ? svld1_f64(npred, C+6*ldc) : svdup_f64(0.0);
    c7x = (m > 7) ? svld1_f64(npred, C+7*ldc) : svdup_f64(0.0);
    c8x = (m > 8) ? svld1_f64(npred, C+8*ldc) : svdup_f64(0.0);
    c9x = (m > 9) ? svld1_f64(npred, C+9*ldc) : svdup_f64(0.0);
    c10x = (m > 10) ? svld1_f64(npred, C+10*ldc) : svdup_f64(0.0);
    c11x = (m > 11) ? svld1_f64(npred, C+11*ldc) : svdup_f64(0.0);
    c12x = (m > 12) ? svld1_f64(npred, C+12*ldc) : svdup_f64(0.0);
    c13x = (m > 13) ? svld1_f64(npred, C+13*ldc) : svdup_f64(0.0);
    c14x = (m > 14) ? svld1_f64(npred, C+14*ldc) : svdup_f64(0.0);
    c15x = (m > 15) ? svld1_f64(npred, C+15*ldc) : svdup_f64(0.0);
    // printf("m: %d n: %d k: %d",m,n,kk);
    for (int kk=0; kk<kc; kk++){
        register float64_t aval = *(A + kk*m + 0);
        ax =svdup_f64(aval);
        bx = svld1_f64(svptrue_b64(), B + kk*n);
        c0x =svmla_f64_m(npred, c0x, bx, ax);
        aval = *(A + kk*m+1);
        ax =svdup_f64(aval);
        c1x =svmla_f64_m(npred, c1x, bx, ax);
        aval = *(A + kk*m+2);
        ax =svdup_f64(aval);
        c2x =svmla_f64_m(npred, c2x, bx, ax);
        aval = *(A + kk*m+3);
        ax =svdup_f64(aval);
        c3x =svmla_f64_m(npred, c3x, bx, ax);
        aval = *(A + kk*m+4);
        ax =svdup_f64(aval);
        c4x =svmla_f64_m(npred, c4x, bx, ax);
        aval = *(A + kk*m+5);
        ax =svdup_f64(aval);
        c5x =svmla_f64_m(npred, c5x, bx, ax);
        aval = *(A + kk*m+6);
        ax =svdup_f64(aval);
        c6x =svmla_f64_m(npred, c6x, bx, ax);
        aval = *(A + kk*m+7);
        ax =svdup_f64(aval);
        c7x =svmla_f64_m(npred, c7x, bx, ax);
        aval = *(A + kk*m+8);
        ax =svdup_f64(aval);
        c8x =svmla_f64_m(npred, c8x, bx, ax);
        aval = *(A + kk*m+9);
        ax =svdup_f64(aval);
        c9x =svmla_f64_m(npred, c9x, bx, ax);
        aval = *(A + kk*m+10);
        ax =svdup_f64(aval);
        c10x =svmla_f64_m(npred, c10x, bx, ax);
        aval = *(A + kk*m+11);
        ax =svdup_f64(aval);
        c11x =svmla_f64_m(npred, c11x, bx, ax);
        aval = *(A + kk*m+12);
        ax =svdup_f64(aval);
        c12x =svmla_f64_m(npred, c12x, bx, ax);
        aval = *(A + kk*m+13);
        ax =svdup_f64(aval);
        c13x =svmla_f64_m(npred, c13x, bx, ax);
        aval = *(A + kk*m+14);
        ax =svdup_f64(aval);
        c14x =svmla_f64_m(npred, c14x, bx, ax);
        aval = *(A + kk*m+15);
        ax =svdup_f64(aval);
        c15x =svmla_f64_m(npred, c15x, bx, ax);
    }
    svst1_f64(npred, C, c0x);
    if (m > 1) svst1_f64(npred, C+ldc, c1x);
    if (m > 2) svst1_f64(npred, C+2*ldc, c2x);
    if (m > 3) svst1_f64(npred, C+3*ldc, c3x);
    if (m > 4) svst1_f64(npred, C+4*ldc, c4x);
    if (m > 5) svst1_f64(npred, C+5*ldc, c5x);
    if (m > 6) svst1_f64(npred, C+6*ldc, c6x);
    if (m > 7) svst1_f64(npred, C+7*ldc, c7x);
    if (m > 8) svst1_f64(npred, C+8*ldc, c8x);
    if (m > 9) svst1_f64(npred, C+9*ldc, c9x);
    if (m > 10) svst1_f64(npred, C+10*ldc, c10x);
    if (m > 11) svst1_f64(npred, C+11*ldc, c11x);
    if (m > 12) svst1_f64(npred, C+12*ldc, c12x);
    if (m > 13) svst1_f64(npred, C+13*ldc, c13x);
    if (m > 14) svst1_f64(npred, C+14*ldc, c14x);
    if (m > 15) svst1_f64(npred, C+15*ldc, c15x);

}


