#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
 char *filename = "test.txt";
 if (access(filename,X_OK)==-1) {
  printf("\nUser cannot read file %s \n", filename);
  exit(1);
 }
 printf("\n%s readable, procedding\n", filename);
}
