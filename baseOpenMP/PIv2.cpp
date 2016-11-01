#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include "myTime.cpp"

int main3(int argc, char* argv[])
{
	long long num_steps = 100000000;
	double step;

	clock_t start, stop;
    double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
	start = clock();

	#pragma omp parallel for private(x) reduction(+:sum)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum += 4.0/(1.+ x*x);
	}

	pi = sum*step;
   stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	printf("Czas przetwarzania wynosi %f\n",((double)(stop - start)/CLOCKS_PER_SEC));
	return 0;
}

/**
   operacji: 1000000000
   wynik: poprawny
   time: 3.72s

   operacji: 100000000
   time: 0.37s
   */
