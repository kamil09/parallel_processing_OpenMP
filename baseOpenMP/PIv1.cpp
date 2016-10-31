#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "myTime.cpp"

long long num_steps = 100000000;
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

	#pragma omp parallel for private(x)
	for (i=0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		#pragma omp atomic
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
	Podstawowy kod:
	Wynik: poprawny
	operacji: 1000000000
	czas: 0.78s
	(CPU-TIME) 7.079299, WALL-TIME: 7.079753
*/

/**
	1:
	#pragma omp parallel for
	//Niepoprawne, ponieważ zmienna sum jest współdzielona, a nie zapewniono wyłączoności dostępu przy operacji zapisie, zmienna x jest również współdzielona, ale mogłaby być lokalna
	(CPU-TIME) 41.961753, WALL-TIME: 10.627785
	operacji: 1000000000
	czas: 3.8s
	wynik : niepoprawny
*/

/**
	2:
	x jako zmienna lokalna
	wynik: niepoprawny
	operacji: 1000000000
	(CPU-TIME) 23.429805, WALL-TIME: 5.987123
*/

/**
	3:
	Dodanie wyrektywy atomic
	wynik: poprawny
	operacji: 1000000000
	(CPU-TIME) 401.363014, WALL-TIME: 102.931273
*/
