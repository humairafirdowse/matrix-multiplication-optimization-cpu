#include "bl_config.h"
#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    register svfloat64_t ax;
    register svfloat64_t bx;
    register svfloat64_t c0x;
    svbool_t npred = svwhilelt_b64_u64(0, n);
    
    for (int kk=0; kk<k; kk++){
        for(int i=0;i<m;i++){
            register float64_t aval = *(a + kk*m + i);
            c0x = svld1_f64(npred, c+i*ldc);
            ax =svdup_f64(aval);
            bx = svld1_f64(svptrue_b64(), b + kk*n);
            c0x =svmla_f64_m(npred, c0x, bx, ax);
            svst1_f64(npred, c+i*ldc, c0x);
        }

    }
}

// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
