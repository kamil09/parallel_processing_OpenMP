#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>
//Długość lini w pamięci podręcznej

int mainLine(int argc, char* argv[])
{
	long long num_steps = 100000000;
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

   //WERSJA 1:
   /*
   for (i = 0; i < calc_count; i++) {
	   sumTable[i] = 0;
	   sumTable[i + 1] = 0;
	   sum = 0;
	   start = clock();

	   #pragma omp parallel private(x)
	   {
		   int th_num = omp_get_thread_num();
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
   for (i = 0; i<calc_count; i++) std::cout << times[i] << std::endl;
	*/

	
	//WERSJA 2
   
	for (i = 0; i < calc_count; i++) {
		sumTable[i] = 0;
		sumTable[i + 1] = 0;
		  #pragma omp parallel private(start,stop)
		  {
			start = clock();
			  int th_num = omp_get_thread_num();
			  for (int j = 0; j<num_steps; j++)
			  {
				  x = (j + .5)*step;
				  sumTable[i + th_num] += 4.0 / (1. + x*x);
			  }

			  pi = sumTable[i + th_num]*step;
			  stop = clock();
			  times[i + th_num] = ((double)stop - start) / CLOCKS_PER_SEC;
		  }
		  printf("time1: %f,time2: %f\n", times[i], times[i+1]);
	   }
    

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	getchar();
	return 0;
}

/**
WERSJA 2:
time1: 2.795000,time2: 2.637000
time1: 3.695000,time2: 3.713000
time1: 3.730000,time2: 3.730000
time1: 3.693000,time2: 3.700000
time1: 3.190000,time2: 3.230000
time1: 1.832000,time2: 1.904000 XXX
time1: 3.364000,time2: 3.390000
time1: 3.493000,time2: 3.508000
time1: 3.348000,time2: 3.312000
time1: 3.314000,time2: 3.328000
time1: 3.525000,time2: 3.530000
time1: 3.571000,time2: 3.573000
time1: 3.205000,time2: 3.226000
time1: 1.972000,time2: 2.009000 XXX
time1: 3.563000,time2: 3.579000
time1: 3.423000,time2: 3.430000
time1: 3.522000,time2: 3.516000
time1: 3.448000,time2: 3.473000
time1: 3.197000,time2: 3.200000
time1: 2.916000,time2: 2.951000


   WINDOWS:
	1.084
	0.549 XXX
	0.777
	0.749
	0.738
	0.759
	0.689
	0.653
	0.699
	0.48  XXX 
	0.854
	0.799
	0.796
	0.775
	0.804
	0.721
	0.812
	0.562 XXX
	0.702
	0.719

	Długość lini: 8słów, double: 8B => 64B
   
   WERSJA 2:
	
   
   */