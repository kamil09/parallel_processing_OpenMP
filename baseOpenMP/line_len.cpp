#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>

//Długość lini w pamięci podręcznej
void line_len(){

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
			for (int j=0; j<num_steps; j++)
			{
			  x = (j + .5)*step;
			  sumTable[i+ omp_get_thread_num()] += 4.0/(1.+ x*x);
			}
		}	
		pi = step*sumTable[i];
		stop = clock();
		std::cout << ((double)stop - start) / CLOCKS_PER_SEC << std::endl;
	 }

	printf("Wartosc liczby PI wynosi %15.12f\n", pi);

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
}
