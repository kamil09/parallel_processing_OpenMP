#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include "myTime.cpp"
//SUMY CZEŚCIOWE

long long num_steps = 1000000000;
double step;

int main(int argc, char* argv[])
{
	clock_t start, stop;
   double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;

   int threads = 4;
   volatile double sumTable[threads];
   omp_set_num_threads(threads);
   for (i=0;i<threads;i++) sumTable[i]=0;

   start = clock();
   startW = get_wall_time();

   // WERSJA 1:
	// #pragma omp parallel for shared(sumTable) private(x)
   // for (i=0; i<num_steps; i++)
	// {
	// 	x = (i + .5)*step;
   //    sumTable[omp_get_thread_num()] += 4.0/(1.+ x*x);
	// }

   //WERSJA 2:
   #pragma omp parallel private(x)
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
	stopW = get_wall_time();
   stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	printf("Czas przetwarzania wynosi (CPU-TIME) %f, WALL-TIME: %f \n",((double)(stop - start)/CLOCKS_PER_SEC),(double)(stopW - startW));
	return 0;
}

/**
   WERSJA 1:
   operacji: 1000000000
   wątki 2:    (CPU-TIME) 10.895711, WALL-TIME: 5.552993
   wątki 4:    (CPU-TIME) 18.198092, WALL-TIME: 5.353006
   wątki 8:    (CPU-TIME) 18.631670, WALL-TIME: 4.728367
   wątki 32:   (CPU-TIME) 16.063877, WALL-TIME: 4.202447
   wątki 1024: (CPU-TIME) 15.315567, WALL-TIME: 3.908528
*/

/**
   WERSJA 2:
   wątki 2:    (CPU-TIME) 9.189632, WALL-TIME: 4.927024
   wątki 4:    (CPU-TIME) 15.322318, WALL-TIME: 3.923391
   wątki 8:    (CPU-TIME) 15.537506, WALL-TIME: 3.931437
   wątki 32:   (CPU-TIME) 15.410359, WALL-TIME: 3.903900
   wątki 1024: (CPU-TIME) 14.456735, WALL-TIME: 3.914390

*/
