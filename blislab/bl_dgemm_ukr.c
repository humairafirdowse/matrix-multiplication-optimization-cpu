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
    
    // for ( l = 0; l < k; ++l )
    // {                 
    //     for ( i = 0; i < m; ++i  )
    //     { 
    //         for ( j = 0; j < n; ++j )
    //         { 
	//       // ldc is used here because a[] and b[] are not packed by the
	//       // starter code
	//       // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	//       //
	//       c( i, j, ldc ) += a( i, l, DGEMM_MR) * b( j, l, DGEMM_NR );   
    //         }
    //     }
    // }
    // printf("%s\n","c: ");
    // bl_printmatrix(c,ldc,m,n);
    // // printf("%s\n","a: ");
    // // bl_printmatrix(a,DGEMM_MR,m,k);
    // printf("%s\n %lf","b: ",b);
    // bl_printmatrix(b,DGEMM_MR,k,n);
    for ( l = 0; l < k; ++l )
    {                 
        for ( i = 0; i < m; ++i )
        { 
            for ( j = 0; j < n; ++j )
            { 
                // printf("values %d:%d:%d",m,n,k);
                c( i, j, ldc ) += a( l, i, m) * b( l, j, n);  
                // printf("a: %lf\n", a(i, l, DGEMM_MR));
                // printf("b: %lf\n", b(l, j, DGEMM_MR));

            }
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
