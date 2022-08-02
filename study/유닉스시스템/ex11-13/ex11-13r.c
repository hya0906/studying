#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define MSGSIZE 64

int main()
{
 char msg[MSGSIZE];
 int filedes;
 int nread, cnt;

 if (mkfifo("./fifo", 0666) == -1) {
  printf("\nFail to call fifo()\n");
  exit(1);
 }
 if ((filedes = open("./fifo", O_RDWR))<0) {
  printf("\nFail to call open()\n");
  exit(1);
 }
 for(cnt=0;cnt<3;cnt++) {
  if((nread = read(filedes,msg,MSGSIZE))<0) {
   printf("/nFail to call read()\n");
   exit(1);
  }
  printf("\nrecv: %s\n", msg);
 }
 unlink("./fifo");
}
