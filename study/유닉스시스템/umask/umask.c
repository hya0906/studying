#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(){
int filedesc;
mode_t oldmask;

oldmask = umask(023);
filedesc = open("text.txt",O_CREAT,0777);
close(filedesc);

}//end ma
