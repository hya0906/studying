#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
 int fd;
 fd = open("data.txt", O_RDWR);
 if(fd<0)
 {
  printf("\nFile open error\n");
  exit(1);
 }
 //not on erro
 printf("\nfd = %d\n",fd);
 
 close(fd);
}
