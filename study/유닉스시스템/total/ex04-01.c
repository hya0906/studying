#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

int main()
{
char *originalname="test.txt";
char *hardfilename="test.txt.hard";
char *softfilename="test.txt.soft";

int filedes, retval;
mode_t oldmask;
char buffer[1024];
int nread;
struct stat finfo;

oldmask = umask(0377); //011 111 111
                       //100 000 000 ->r-- --- ---

filedes = open(originalname, O_RDWR|O_CREAT, 0755); // r-- --- ---(0400)
close(filedes);

if((retval=access(originalname, W_OK))==-1){
  printf("\n%s is not writible\n" , originalname);
  chmod(originalname, 0644);//0400 -> 0644
}

link(originalname, hardfilename);
symlink(originalname, softfilename);

rename(hardfilename,"newname.txt");

nread = readlink(softfilename, buffer, 1024);
write(1,buffer,nread);

stat(originalname, &finfo);
printf("\n%s", originalname);
printf("\nFile Mode: %o", finfo.st_mode);
printf("\nFile Size: %ld", finfo.st_size);
printf("\nNo of Blocks: &ld\n", finfo.st_blocks);
}
