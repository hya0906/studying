#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

#define MSGSIZE 64

int main()
{
  char msg[MSGSIZE];
  int filedes;
  int cnt;

  if (( filedes = open("./fifo", O_WRONLY))<0) {
   printf("\nFail to call open()\n");
   exit(1);
  }
  for ( cnt=0;cnt<3;cnt++) {
   printf("\nInput a message: ");
   scanf("%s", msg);

   if(write(filedes, msg, MSGSIZE) == -1) {
    printf("\nFail to call write()\n");
    exit(1);
   }
   sleep(1);
  }
}
