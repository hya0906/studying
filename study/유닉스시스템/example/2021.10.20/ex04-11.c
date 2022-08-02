#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
 struct stat finfo;
 char fname[1024];
 
 if(argc>1){
  strcpy(fname, argv[1]);
 }
 else{
  strcpy(fname, argv[0]);
 }

 if(stat(fname, &finfo) == -1) {
  fprintf(stderr, "Couldn't stat %s \n", fname);
  exit(1)
 }
 
 printf("%s \n", fname);
 printf("ID of device: %d \n", finfo.st_dev);
 printf("Inode number: %d \n", finfo.st_ino);
 printf("File mode: %o \n", finfo.st_mode);
 printf("Num of links: %d \n", finfo.st_nlink);
 printf("User ID: %d \n", finfo.st_uid);
 printf("Group ID: %d \n", finfo.st_gid);
 printf("Files size: %d \n", finfo.st_size);
 printf("Last access time: %u \n", finfo.st_atim);
 printf("Last modify time: %u \n", finfo.st_mtim);
 printf("Last stat change: %u \n", finfo.st_ctim);
 printf("I/O Block size: %d \n", finfo.st_blksize);
 printf("Num of blocks: %d \n", finfo.st_blocks);
 printf("File system: %s \n", finfo.st_fstype);
}
