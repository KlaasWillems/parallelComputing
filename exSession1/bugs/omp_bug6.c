/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   Fails compilation in most cases.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];
float localsum;

float dotprod(float *sum)
{
  int i, tid;

  tid = omp_get_thread_num();
  localsum = 0.0;
  #pragma omp for reduction(+ : localsum) 
  for (i = 0; i < VECLEN; i++)
    {
      localsum = localsum + (a[i] * b[i]);
      printf("  tid= %d i=%d\n", tid, i);
    }
  *sum = *sum + localsum;
  return localsum;
}

int main(int argc, char *argv[])
{
  int i;
  float sum;
  
  for (i = 0; i < VECLEN; i++)
    a[i] = b[i] = 1.0 * i;
  sum = 0.0;

  dotprod(&sum);

  printf("Sum = %f\n", sum);
}
