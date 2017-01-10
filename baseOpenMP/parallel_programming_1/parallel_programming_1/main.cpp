#include <stdio.h>
#include <iostream>
#include <time.h>
#include <omp.h>
#include "line_len.h"
#include "pi3.h"
#include "pi2.h"
#include "pi1.h"


int main(int argc, char** argv) {
	puts("\nSEKWENCYJNE");
	pi1_sek();
	puts("\nPRAGMA OMP PARALLEL");
	pi1_paral();
	puts("\n+PRIVATE X");
	pi1_paral_x_priv();
	puts("\nATOMIC");
	pi1_paral_x_priv_atomic();
	puts("\nREDUCTION");
	pi2();
	puts("\n SHARED");
	pi3();
	puts("\n LINE LEN");
	line_len();
	return 0;
}
/*

SEKWENCYJNE
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 0.719000

PRAGMA OMP PARALLEL
Wartosc liczby PI wynosi  1.221907672033
Czas przetwarzania wynosi 0.541000

+PRIVATE X
Wartosc liczby PI wynosi  1.066556999334
Czas przetwarzania wynosi 0.509000

ATOMIC
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 10.128000

REDUCTION
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 0.372000

SHARED
Wartosc liczby PI wynosi  3.141592653590
Czas przetwarzania wynosi 0.488000s

LINE LEN
2.293
2.399
2.43
2.438
1.333
2.569
2.585
2.54
2.539
2.543
2.548
2.472
1.354
2.471
2.454
2.454
2.498
2.544
2.506
2.445
Wartosc liczby PI wynosi  3.141592653590
*/