#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <stdlib.h> // For: exit, random, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>
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
    int l, j, i;
    

    for ( l = 0; l < k; ++l )
    {                 
        for ( i = 0; i < m; ++i )
        { 
            for ( j = 0; j < n; j+=1)
            { 
                c( i, j, ldc ) += a( l, i, m) * b( l, j, n) ;

            }
        }
    }
    // for ( l = 0; l < k; ++l )
    // {                 
    //     for ( j = 0; j < n; j+=4 )
    //     { 
    //         blj = b( l, j, n); 
    //         for ( i = 0; i < m; ++i )
    //         { 
    //             c( i, j, ldc ) += a( l, i, m) * blj 

    //         }
    //     }
    // }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
