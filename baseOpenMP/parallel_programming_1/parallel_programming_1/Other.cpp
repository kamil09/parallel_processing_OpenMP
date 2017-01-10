#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <windows.h>

int main5() {
	int i;
	int l_proc = omp_get_num_procs();
	omp_set_num_threads(3);
	printf("Proc num: %d\n", l_proc);

	#pragma omp parallel 
	{
		/*
		//#pragma omp for schedule(static,2)
		#pragma omp for schedule(dynamic,2) 
		for (i = 0; i < 20; i++) {
			printf("Iter:%d wykonal watek %d \n", i, omp_get_thread_num());
		
		}
		*/
		int th_id = omp_get_thread_num();
		HANDLE thread_handler = GetCurrentThread();
		//otrzymanie w³asnego identyfikatora
		DWORD_PTR mask = (1 << (th_id % l_proc));
		//okreœlenie maski dla przetwarzania w¹tku wy³¹cznie na jednym procesorzeprzydzielanym modulo liczba procesorów
		DWORD_PTR result = SetThreadAffinityMask(thread_handler, mask);
		//przekazanie do systemu operacyjnego maski powinowactwa
		if (result == 0) printf("blad SetThreadAffnityMask \n");
		else
		{
			printf("maska poprzednia dla watku %d : %d\n", th_id, result);
			printf("maska nowa dla watku %d : %d\n", th_id, SetThreadAffinityMask(
				thread_handler, mask));
		}

	}
	return 0;
}