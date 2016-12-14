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
		sum = 0;
		start = clock();

		#pragma omp parallel private(x) shared(sumTable)
		{
			int th_num = omp_get_thread_num();
			HANDLE thread_handler = GetCurrentThread();
			DWORD_PTR mask = (1 << (th_num % 2));

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
	for (i = 0; i<calc_count; i++) std::cout << times[i] << std::endl;
	 


	 //WERSJA 2
	/*
	for (i = 0; i < calc_count; i++) {
		sumTable[i] = 0;
		sumTable[i + 1] = 0;
#pragma omp parallel private(start,stop) shared(sumTable)
		{
			start = clock();
			int th_num = omp_get_thread_num();

			for (int j = 0; j < num_steps; j++)
			{
				x = (j + .5)*step;
				sumTable[i + th_num] += 4.0 / (1. + x*x);
			}

			pi = sumTable[i + th_num] * step;
			stop = clock();
			times[i + th_num] = ((double)stop - start) / CLOCKS_PER_SEC;
		}
		printf("time1: %f,time2: %f\n", times[i], times[i + 1]);
	}*/


	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	getchar();
	return 0;


	/**
	WERSJA 2:
	time1: 3.756000,time2: 3.787000
	time1: 3.554000,time2: 3.557000
	time1: 3.445000,time2: 3.478000
	time1: 3.384000,time2: 3.426000
	time1: 3.524000,time2: 3.517000
	time1: 1.813000,time2: 1.840000 XXX
	time1: 3.521000,time2: 3.539000
	time1: 3.538000,time2: 3.569000
	time1: 3.562000,time2: 3.602000
	time1: 3.430000,time2: 3.439000
	time1: 3.777000,time2: 3.772000
	time1: 3.776000,time2: 3.768000
	time1: 3.229000,time2: 3.240000
	time1: 1.827000,time2: 1.749000 XXX
	time1: 3.687000,time2: 3.658000
	time1: 3.306000,time2: 3.272000
	time1: 3.801000,time2: 3.803000
	time1: 3.648000,time2: 3.662000
	time1: 3.700000,time2: 3.704000
	time1: 3.566000,time2: 3.568000

	WERSJA 1:
		1.063
		0.759 XXX
		1.048
		1.046
		1.057
		1.046
		1.057
		1.05
		1.045
		0.756 XXX
		1.042
		1.05
		1.051
		1.043
		1.053
		1.05
		1.034
		0.756 XXX
		1.05
		1.046

		Długość lini: 8słów, double: 8B => 64B*/
}
