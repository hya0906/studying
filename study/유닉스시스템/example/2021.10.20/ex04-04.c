#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
 char *filename = "test.txt";
 if ( access(filename, X_OK) == -1){
  fprintf(stderr, "User cannot execute file %s \n", filename);
  exit(1);
 }
 printf("%s executable proceeding \n", filename);
}
