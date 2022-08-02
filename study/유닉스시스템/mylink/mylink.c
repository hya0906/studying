#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//mylink
//$ ./mylink a1 a2 a3
//in mylink.c
//argc: count of argv
//argv[0] : ./mylink
//argv[1] : a1
//argv[2] : a2
//argv[3] : a3
//$ cat test.txt

int main(int argc, char *argv[])
{
 //printf("\nargc = %d",argc);
 //printf("\nargv[0] = %s", argv[0]);
 //printf("\nargv[1] = %s", argv[1]);
 //printf("\nargv[2] = %s", argv[2]);
 //printf("\nargv[3] = %s", argv[3]);
 //printf("\n");
 
 if(link(argv[1], argv[2]))
   printf("\nhard link failed\n");
}
