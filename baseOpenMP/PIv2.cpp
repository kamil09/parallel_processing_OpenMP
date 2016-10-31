#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include "myTime.cpp"

long long num_steps = 1000000000;
double step;

int main(int argc, char* argv[])
{
	clock_t start, stop;
   double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
	start = clock();
   startW = get_wall_time();

	#pragma omp parallel for private(x) reduction(+:sum)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum += 4.0/(1.+ x*x);
	}

	pi = sum*step;
	stopW = get_wall_time();
   stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	printf("Czas przetwarzania wynosi (CPU-TIME) %f, WALL-TIME: %f \n",((double)(stop - start)/CLOCKS_PER_SEC),(double)(stopW - startW));
	return 0;
}

/**
   operacji: 1000000000
   (CPU-TIME) 14.848310s, WALL-TIME: 3.766263s
   wynik: poprawny
*/
