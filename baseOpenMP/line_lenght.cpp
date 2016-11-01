#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>
//Długość lini w pamięci podręcznej

int mainx(int argc, char* argv[])
{
	long long num_steps = 200000000;
	double step;

	time_t start, stop;
	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;

	const int threads = 2;
   const int calc_count = 20;
   double times[calc_count];
   volatile double sumTable[calc_count+2];
   omp_set_num_threads(2);

   for (i=0;i<calc_count;i++){
      sumTable[i]=0;
      sumTable[i+1]=0;
      sum=0;
	  start = clock();

      #pragma omp parallel private(x)
      {
		  int th_num = omp_get_thread_num();
		  HANDLE thread_handler = GetCurrentThread();
		  DWORD_PTR mask = (1 << (th_num % threads));
		  DWORD_PTR result = SetThreadAffinityMask(thread_handler, mask);

         #pragma omp for schedule(static,1)
         for (int j=0; j<num_steps; j++)
         {
            x = (j + .5)*step;
            sumTable[i+th_num] += 4.0/(1.+ x*x);
         }
      }

      sum = sumTable[i]+sumTable[i+1];

      pi = sum*step;
	  stop = clock();
      times[i] = ((double)stop-start)/CLOCKS_PER_SEC;
   }

   for (i=0;i<calc_count;i++) std::cout << times[i] << std::endl;

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	getchar();
	return 0;
}

/**
   WINDOWS: (wykorzystując słowa od siebie oddalone o 1 na wypadek gdyby krawędź lini wypadła w środku słowa)
    1.455 XX
	1.445 XX
	2.067
	2.018
	2.023
	2.052
	2.028
	2.028
	1.447 XX
	1.446 XX
	2.069
	2.012
	2.034
	2.054
	2.027
	2.029
	1.448 XX
	1.447 XX
	2.061
	2.019

   WINDOWS:
	2.011
	1.449 XXX
	1.986
	2.042
	1.964
	1.982
	1.983
	1.981
	1.978
	1.447 XXX
	1.984
	2.055
	1.958
	1.98
	1.97
	1.974
	1.99
	1.45 XXX
	1.984
	2.067

	Długość lini: 8słów, double: 8B => 64B
   */