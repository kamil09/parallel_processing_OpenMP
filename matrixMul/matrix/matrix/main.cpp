/*
	TEMAT 2:
	Mno�enie macierzy por�wnanie efektywno�ci metod �
		3 p�tle - kolejno�� p�tli: ijk,
		6 p�tli - kolejno�� p�tli: zewn�trznych ijk, wewn�trznych: ikj.
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include "omp.h"
#include <cmath>

static const int ROWS = 1000;     // liczba wierszy macierzy
static const int COLUMNS = 1000;  // lizba kolumn macierzy
float matrix_a[ROWS][COLUMNS];    // lewy operand 
float matrix_b[ROWS][COLUMNS];    // prawy operand
float matrix_r[ROWS][COLUMNS];    // wynik
float matrix_r1[ROWS][COLUMNS];

FILE *result_file;

void initialize_matrices();
void clear_result_matrix();
void print_elapsed_time(double start, char* description);
/*
	Spr�bujmy opisa� co mo�e by� przyczyn� takiego a nie innego czasu przetwarania
*/
void multiply_matrices_IJK();
/*
	To samo co wy�ej tylko w 6 p�tlach
*/
void multiply_matrices_IJK_IKJ();


int main(int argc, char* argv[]) {
	if ((result_file = fopen("classic.txt", "a")) == NULL) {
		fprintf(stderr, "nie mozna otworzyc pliku wyniku \n");
		perror("classic");
		return(EXIT_FAILURE);
	}

	double start = clock() / CLK_TCK;
	initialize_matrices();
	print_elapsed_time(start, "Inicjalizacja macierzy");

	clear_result_matrix();
	start = clock() / CLK_TCK;
	multiply_matrices_IJK();
	print_elapsed_time(start, "Mno�enie macierzy 3 p�tle IJK");

	//ONLY TO TEST IF 6LOOP method is correct
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_r1[i][j] = matrix_r[i][j];
		}
	}

	clear_result_matrix();
	start = clock() / CLK_TCK;
	multiply_matrices_IJK_IKJ();
	print_elapsed_time(start, "Mno�enie macierzy 6 p�tli IJK-IKJ");

	//ONLY TO TEST IF 6LOOP method is correct
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			if (abs(matrix_r1[i][j] - matrix_r[i][j]) > 0.001  ) {
				printf("%f != %f\n", matrix_r1[i][j], matrix_r[i][j]);
			}
		}
	}
	
	fclose(result_file);
	return 0;
}


void multiply_matrices_IJK()
{ 
	#pragma omp parallel for
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0 ; j < COLUMNS; j++) {
			float sum = 0.0;
			for (int k = 0 ; k < COLUMNS; k++)
				sum += matrix_a[i][k] * matrix_b[k][j];
			matrix_r[i][j] = sum;
		}
	}
}

void multiply_matrices_IJK_IKJ()
{
	//DZIA�A pmi�ta�, �e ROWS,COLUMNS %r musi == 0  (musi by� mo�liwo�� podziau macierzy na grupy)
	int r = 10;
	#pragma omp parallel for
	for (int i = 0; i < ROWS; i+=r) {
		for (int j = 0; j < COLUMNS; j+=r) {
			for (int k = 0; k < COLUMNS; k+=r) {
				for (int ii = i; ii < i + r; ii++) { // WYNIK CZʌCIOWY
					for (int kk = k; kk < k + r; kk++) {
						for (int jj = j; jj < j + r; jj++) {
							matrix_r[ii][jj] += matrix_a[ii][kk] * matrix_b[kk][jj];
						}
					}
				}
			}
		}
	}
}

void initialize_matrices()
{
	//#pragma omp parallel for 
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLUMNS; j++) {
			matrix_a[i][j] = (float)rand() / RAND_MAX;
			matrix_b[i][j] = (float)rand() / RAND_MAX;
			matrix_r[i][j] = 0.0;
		}
	}
}

void clear_result_matrix()
{
	#pragma omp parallel for 
	for (int i = 0; i < ROWS; i++)
		for (int j = 0; j < COLUMNS; j++)
			matrix_r[i][j] = 0.0;
}

void print_elapsed_time(double start, char* description)
{
	double elapsed;

	// wyznaczenie i zapisanie czasu przetwarzania
	elapsed = (double)clock() / CLK_TCK;
	printf("%s \t\t| czas: %8.4f sec \n", description, elapsed - start);
	fprintf(result_file,
		"%s \t\t| czas : %8.4f sec (%6.4f sec rozdzielczosc pomiaru)\n",
		description, elapsed-start, 1.0 / CLK_TCK);
}