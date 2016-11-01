#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include "myTime.cpp"
//SUMY CZEŚCIOWE

int main(int argc, char* argv[])
{
	long long num_steps = 100000000;
	double step;

	clock_t start, stop;
    double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;

   const int threads = 16;
   volatile double sumTable[threads];
   omp_set_num_threads(threads);
   for (i=0;i<threads;i++) sumTable[i]=0;

   start = clock();

   // WERSJA 1:
   /*
	 #pragma omp parallel for shared(sumTable) private(x)
    for (i=0; i<num_steps; i++)
    {
	 	x = (i + .5)*step;
        sumTable[omp_get_thread_num()] += 4.0/(1.+ x*x);
    }*/

   //WERSJA 2:
   #pragma omp parallel shared(sumTable) private(x)
   {
      int th_num = omp_get_thread_num();
      #pragma omp for
      for (i=0; i<num_steps; i++)
      {
         x = (i + .5)*step;
         sumTable[th_num] += 4.0/(1.+ x*x);
      }

   }

   #pragma omp parallel for reduction(+:sum)
   for (i=0;i<threads;i++) sum += sumTable[i];

   pi = sum*step;
   stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	printf("Czas przetwarzania wynosi %fs\n",((double)(stop - start)/CLOCKS_PER_SEC));
	return 0;
}

/**
   WERSJA 1:
   operacji: 100000000
   wątki 2 : 1.39s
   wątki 4 : 1.33s
   wątki 8 : 1.16s
   wątki 16: 0.86s
   */

/**
   WERSJA 2:
   wątki 2 : 0.50s
   wątki 4 : 0.50s
   wątki 8 : 0.51s
   wątki 16: 0.49s
*/
