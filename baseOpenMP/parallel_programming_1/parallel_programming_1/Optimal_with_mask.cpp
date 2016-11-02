#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>

int main(int argc, char* argv[])
{
	long long num_steps = 100000000;
	const int thread = 4;
	const int procs_num = omp_get_num_procs();
	double step;
	omp_set_num_threads(thread);

	clock_t start, stop;
	double startW = 0, stopW = 0;

	double x, pi, sum = 0.0;
	int i;
	step = 1. / (double)num_steps;
	start = clock();
	#pragma omp parallel private(x) reduction(+:sum)
	{
		int th_num = omp_get_thread_num();
		HANDLE thread_handler = GetCurrentThread();
		DWORD_PTR mask = (1 << (th_num/2 % procs_num));
		
		DWORD_PTR result = SetThreadAffinityMask(thread_handler, mask);
		#pragma omp for
		for (i = 0; i < num_steps; i++)
		{
			x = (i + .5)*step;
			sum += 4.0 / (1. + x*x);
		}
	}
	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f\n", ((double)(stop - start) / CLOCKS_PER_SEC));
	return 0;
}

/**
NO MASK:
	threads: 4
	time: 0.392000

	threads: 8
	time: 0.38800

Po 2 do 1 rdzenia
	threads: 4
	time: 0.744000

	threads: 8
	time: 0.414000

1 do 1
	threads: 4
	time: 0.428000

	threads: 8
	time: 0.411000

*/

