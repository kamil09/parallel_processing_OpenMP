#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>
//Długość lini w pamięci podręcznej

int main(int argc, char* argv[])
{
	long long num_steps = 100000000;
	double step;

	time_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	step = 1. / (double)num_steps;

	const int threads = 2;
	const int calc_count = 20;
	double times[calc_count];
	volatile double sumTable[calc_count + 2];
	omp_set_num_threads(2);

	//WERSJA 1:
	
	for (i = 0; i < calc_count; i++) {
		sumTable[i] = 0;
		sumTable[i + 1] = 0;
		start = clock();

		#pragma omp parallel private(x) shared(sumTable)
		{
			int th_num = omp_get_thread_num();
			for (int j=0; j<num_steps; j++)
			{
			  x = (j + .5)*step;
			  sumTable[i+ th_num] += 4.0/(1.+ x*x);
			}
		}	
		pi = step*sumTable[i];
		stop = clock();
		std::cout << ((double)stop - start) / CLOCKS_PER_SEC << std::endl;
	 }

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	getchar();
	return 0;

	/*
	2.563
	2.746
	2.479
	2.488
	2.426
	1.298
	2.586
	2.596
	2.585
	2.616
	2.598
	2.618
	2.578
	1.465
	2.43
	2.455
	2.409
	2.402
	2.423
	2.398
		
	Wartosc liczby PI wynosi  3.141592653590
	*/
	//Długość lini: 8słów, double: 8B => 64B*/


	/*
	1.53
	1.744
	1.527
	1.627
	1.542
	0.838
	1.402
	1.35
	1.267
	1.198
	1.268
	1.321
	1.23
	0.855
	1.213
	1.236
	1.306
	1.212
	1.19
	1.249
	*/
}
