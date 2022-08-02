#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
 DIR *dirp;
 if((dirp = opendir("test_dir1")) == NULL)
 {
  fprintf(stderr, "Error on opening directory test_dir1\n");
  exit(1);
 }
 printf("success!");
 closedir(dirp);
}
