#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int main() {
	omp_set_num_threads(3);

#pragma omp parallel
	{
		int th_id = omp_get_thread_num();
		HANDLE thread_uchwyt = GetCurrentThread();
		//otrzymanie w�asnego identyfikatora
		DWORD_PTR mask = (1 << (th_id % omp_get_num_procs()));
		//okre�lenie maski dla przetwarzania w�tku wy��cznie na jednym procesorze przydzielanym modulo liczba procesor�w
		DWORD_PTR result = SetThreadAffinityMask(thread_uchwyt, mask);
		//przekazanie do systemu operacyjnego maski powinowactwa
		if (result == 0) printf("blad SetThreadAffnityMask \n");
		else
		{
			printf("maska poprzednia dla watku %d : %d\n", th_id, result);
			printf("maska nowa dla watku %d : %d\n", th_id, SetThreadAffinityMask(thread_uchwyt, mask));
      }
	}
}
