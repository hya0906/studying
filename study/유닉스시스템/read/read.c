/* we implement a part of cat command*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main()
{
 int fd1;
 ssize_t nread;
 char buffer[1024];
 int total_chars=0;

 fd1 = open("temp1.txt", O_RDONLY);
 if (fd1<0) {
  printf("\nFile Open Error\n");
  exit(1);
 }// end if

 while ((nread = read(fd1,buffer,1024))>0) {
  total_chars += nread;
  printf("%s", buffer);
 } // end while
 
 printf("\n\nTotal Chars:%d\n", total_chars);
 close(fd1);
} // end main
