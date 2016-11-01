#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "myTime.cpp"

int main2(int argc, char* argv[])
{
	long long num_steps = 100000000;
	double step;

	clock_t start, stop;
    double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;
	start = clock();

	#pragma omp parallel for private(x)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		#pragma omp atomic
		sum += 4.0/(1.+ x*x);
	}

	pi = sum*step;
    stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	printf("Czas przetwarzania wynosi %f\n",((double)(stop - start)/CLOCKS_PER_SEC));
	return 0;
}

/**
	Podstawowy kod:
	Wynik: poprawny
	operacji: 100000000
	time: 0.72s
*/

/**
	1:
	#pragma omp parallel for
	//Niepoprawne, ponieważ zmienna sum jest współdzielona, a nie zapewniono wyłączoności dostępu przy operacji zapisie, zmienna x jest również współdzielona, ale mogłaby być lokalna
	operacji: 100000000
	wynik : niepoprawny
	time: 2.13s
*/

/**
	2:
	x jako zmienna lokalna
	wynik: niepoprawny
	operacji: 100000000
	time: 0.52s
*/

/**
	3:
	Dodanie wyrektywy atomic
	wynik: poprawny
	operacji: 100000000
	time: 9.196000s
*/
