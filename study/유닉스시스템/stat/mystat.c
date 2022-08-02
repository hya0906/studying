#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// $ ./mystat <filename>
//argv[0] <- ./mystat
//argv[1] <- <filename>

int main(int argc, char *argv[])
{
struct stat finfo; //declaration
char fname[1024];

if (argc>1)
   strcpy(fname, argv[1]);
else
   strcpy(fname, argv[0]);

if (stat(fname, &finfo) == -1) {
   printf("\nCouldn't stat %s \n", fname);
   exit(1); //exit status <- -1
 }

printf("\nstat of %s", fname);
printf("\nID of device: %ld", finfo.st_dev);
printf("\ni-node number: %ld", finfo.st_ino);
printf("\nFile Mode: %o", finfo.st_mode);
printf("\nNo of links: %ld", finfo.st_nlink);
printf("\nUser ID: %d", finfo.st_uid);
printf("\nGroup ID: %d", finfo.st_gid);
printf("\nFiles Size: %ld", finfo.st_size);
printf("\nLast Access Time: %u",  finfo.st_atim);
printf("\nLast Modify Time: %u",  finfo.st_mtim);
printf("\nLast Change Time: %u",  finfo.st_ctim);
printf("\nI/O Block Size: %ld", finfo.st_blksize);
printf("\nNo of Blocks: %ld", finfo.st_blocks);

}//end main
