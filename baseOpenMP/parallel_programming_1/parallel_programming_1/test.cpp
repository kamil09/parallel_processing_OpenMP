#include "stdafx.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>

int liczPiSekwencyjnie() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	long long num_steps = 100000000;
	double step;
	step = 1. / (double)num_steps;
	start = clock();
	for (i = 0; i<num_steps; i++)
	{
		x = (i + .5)*step;
		sum = sum + 4.0 / (1. + x*x);
	}

	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", ((double)(stop - start) / 1000.0));
	return 0;
}

int liczPiRownolegle() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	long long num_steps = 100000000;
	double step;
	step = 1. / (double)num_steps;
	start = clock();
#pragma omp parallel for
	for (i = 0; i<num_steps; i++) //Prywatna zmienna i
	{
		x = (i + .5)*step; //Wszystkie inne zmienne publiczne
		sum = sum + 4.0 / (1. + x*x);
	}

	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", ((double)(stop - start) / 1000.0));
	return 0;
}


int liczPi1() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	long long num_steps = 100000000;
	double step;
	int i;
	step = 1. / (double)num_steps;
	start = clock();
	// Wykonywanie pêtli for, gdzie wartoœæ x jest prywatna dla ka¿dego w¹tku	
#pragma omp parallel for private(x)
	for (i = 0; i<num_steps; i++)
	{
		x = (i + .5)*step;
#pragma omp atomic
		//atomowoœæ modyfikacji
		//¯eby wczytywanie i zapisywanie ostatecznej wartoœci, by³o wykonywane tylko przez jeden proces
		sum += +4.0 / (1. + x*x);
	}

	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", ((double)(stop - start) / 1000.0));
	return 0;
}

int liczPi2() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	long long num_steps = 100000000;
	double step;
	step = 1. / (double)num_steps;
	start = clock();
	omp_set_num_threads(2);
#pragma omp parallel for reduction(+:sum) private(x)
	for (i = 0; i < num_steps; i++)
	{
		x = (i + .5)*step;
		sum += +4.0 / (1. + x*x);
	}
	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", ((double)(stop - start) / 1000.0));
	return 0;
}


int liczPi3() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	long long num_steps = 100000000;
	double step;
	step = 1. / (double)num_steps;
	start = clock();
	const int liczbaWatkow = 4;
	omp_set_num_threads(liczbaWatkow);
	volatile double tab[liczbaWatkow] = { 0 };
#pragma omp parallel
	{
#pragma omp for private(x)
		for (i = 0; i < num_steps; i++)
		{
			x = (i + .5)*step;
			tab[omp_get_thread_num()] += +4.0 / (1. + x*x);
		}
	}
	for (int j = 0; j < liczbaWatkow; j++) {
		sum += tab[j];
	}
	pi = sum*step;
	stop = clock();

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	printf("Czas przetwarzania wynosi %f sekund\n", ((double)(stop - start) / 1000.0));
	return 0;
}


int dlugoscLinii() {
	clock_t start, stop;
	double x, pi, sum = 0.0;
	int i;
	long long num_steps = 100000000;
	double step;
	step = 1. / (double)num_steps;

	const int liczbaWatkow = 2;
	omp_set_num_threads(liczbaWatkow);
	const int liczbaPowtorzen = 20;
	double tablicaCzasow[liczbaPowtorzen] = { 0 };
	volatile double tab[liczbaPowtorzen + 1] = { 0 };
	for (int j = 0; j < liczbaPowtorzen; j++) {
		start = clock();
#pragma omp parallel
		{
			//#pragma omp for schedule(static,1)
#pragma omp for private(x)
			for (i = 0; i < num_steps; i++)
			{
				x = (i + .5)*step;
				tab[j + omp_get_thread_num()] += +4.0 / (1. + x*x);
			}
		}
		pi = tab[j] * step;
		stop = clock();
		tablicaCzasow[j] = ((double)stop - start) / 1000.0;
		printf("%f\n", tablicaCzasow[j]);
	}
	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	return 0;
}


int main() {
	printf("Liczba Pi sekwencyjnie\n");
	liczPiSekwencyjnie();
	printf("\nLiczba Pi rownolegle\n");
	liczPiRownolegle();
	printf("\nPierwsza wersja PI\n");
	liczPi1();
	printf("\nDruga wersja PI\n");
	liczPi2();
	printf("\nTrzecia wersja PI\n");
	liczPi3();
	printf("Dlugosc linii\n");
	dlugoscLinii();
	//getchar();
}



int mainx(int argc, char* argv[])
{
	long long num_steps = 200000000;
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

	for (i = 0; i<calc_count; i++) {
		sumTable[i] = 0;
		sumTable[i + 1] = 0;
		sum = 0;
		start = clock();

#pragma omp parallel private(x)
		{
			int th_num = omp_get_thread_num();
			HANDLE thread_handler = GetCurrentThread();
			DWORD_PTR mask = (1 << (th_num % threads));
			DWORD_PTR result = SetThreadAffinityMask(thread_handler, mask);

#pragma omp for schedule(static,1)
			for (int j = 0; j<num_steps; j++)
			{
				x = (j + .5)*step;
				sumTable[i + th_num] += 4.0 / (1. + x*x);
			}
		}

		sum = sumTable[i] + sumTable[i + 1];

		pi = sum*step;
		stop = clock();
		times[i] = ((double)stop - start) / CLOCKS_PER_SEC;
	}

	for (i = 0; i<calc_count; i++) std::cout << times[i] << std::endl;

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);
	getchar();
	return 0;
}
//Wyniki
/*Liczba Pi sekwencyjnie
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 0.581000 sekund

Liczba Pi rownolegle
Wartosc liczby PI wynosi  0.994680835203
Czas przetwarzania wynosi 1.433000 sekund

Pierwsza wersja PI
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 9.954000 sekund

Druga wersja PI
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 0.346000 sekund

Trzecia wersja PI
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 1.373000 sekund
Dlugosc linii
1.436000
0.603000 Pocz¹tek nowej linii
1.280000
1.290000
1.368000
1.365000
1.364000
1.450000
1.158000
0.597000 Pocz¹tek nowej linii
1.367000
1.335000
1.407000
1.381000
1.390000
1.389000
1.310000
0.624000 Pocz¹tek nowej linii
1.241000
1.249000
Wartosc liczby PI wynosi  3.141592653590

Linia 8 elementów tablicy * 8 bajtów (bo tablica double) = 64 bajty
*/