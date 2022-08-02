#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUF_SIZE 256

void printcwd(void)
{
 char buffer[BUF_SIZE];

 if (getcwd(buffer, BUF_SIZE) == NULL){
  printf("\nError getcwd()");
  exit(1);
 }//end if()
 printf("\n%s \n",buffer);
}//end printcwd()

int main()
{
 printcwd();
 chdir("/usr/include");
 printcwd();
 chdir("..");
 printcwd();
}

