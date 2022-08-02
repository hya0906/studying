#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

int main()
{
 mode_t mode1, mode2;
 mode1 = S_IRUSR| S_IWUSR | S_IRGRP| S_IROTH;
 mode2 = 0644;

 if(chmod("test1.txt", mode1) == -1) {
   printf("\nchmod error\n");
   exit(1);
  }

 if(chmod("test2.txt", mode2) == -1) {
   printf("\nchmod error\n");
   exit(1);
  }
 
 printf("rest of program... \n");
}
