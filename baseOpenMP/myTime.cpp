#include <time.h>
#include <sys/time.h>

double get_wall_time(){
   struct timeval time;

   if (gettimeofday(&time,NULL)){
      return 0;
   }
   return (double)time.tv_sec + ((double)time.tv_usec/1000000);
}
