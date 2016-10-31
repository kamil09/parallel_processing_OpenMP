#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include "myTime.cpp"
//Długość lini w pamięci podręcznej

long long num_steps = 200000000;
double step;

int main(int argc, char* argv[])
{
   double startW=0, stopW=0;

	double x, pi, sum=0.0;
	int i;
	step = 1./(double)num_steps;

   int calc_count = 20;
   double times[calc_count];
   volatile double sumTable[calc_count+2];
   omp_set_num_threads(2);

   for (i=0;i<calc_count;i++){
      startW = get_wall_time();
      sumTable[i]=0;
      sumTable[i+1]=0;
      sum=0;

      //WERSJA 2:
      #pragma omp parallel private(x)
      {
         int th_num = omp_get_thread_num();
         #pragma omp for
         for (int j=0; j<num_steps; j++)
         {
            x = (j + .5)*step;
            if(th_num>1) printf("dddd" );
            sumTable[i+th_num] += 4.0/(1.+ x*x);
         }
      }

      sum = sumTable[i]+sumTable[i+1];

      pi = sum*step;
   	stopW = get_wall_time();
      times[i] = stopW-startW;
   }

   for (i=0;i<calc_count;i++) std::cout << times[i] << std::endl;

	printf("Wartosc liczby PI wynosi %15.12f\n",pi);
	return 0;
}

/**
   WYNIK:
   1.03107
   0.956747
   0.959468
   0.953311
   0.958599
   0.965583
   0.957771
   0.776959 XXX
   0.933827
   0.939766
   0.948542
   0.934803
   0.939504
   0.948353
   0.931035
   0.792129 XXX
   0.961498
   0.957735
   0.960245
   0.950946

   Ilość słów na lini: 8, double: 8B => długość lini to 64B
*/
