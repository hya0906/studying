#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BUF_SIZE 256

void printcwd(void)
{
 char buffer[BUF_SIZE];

 if (getcwd(buffer, BUF_SIZE)==NULL) {
  printf("\nERROR getcwd()");
  exit(1); // 1 means error
 }
 printf("\n%s \n", buffer);
} // end printcwd()

int main()
{
 printcwd(); //print current directory
 chdir("/usr/include"); //move to this dir
 printcwd(); //confirm if dir has changed
 chdir(".."); //move to previous directory
 printcwd();

}






