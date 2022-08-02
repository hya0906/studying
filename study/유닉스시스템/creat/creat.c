/* This file is to compare create() with open()
 * Author: Lecturer
 * Date: 2021.10.17
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main()
{
 int fd1, fd2;
 
 //fd = open("data.txt", O_RDWR);
 fd1 = open("data1.txt", O_WRONLY|O_CREAT|O_TRUNC, 0644); //same with creat
 if (fd1<0) {
    printf("\nFile open error\n");
    exit(1);
 }

 //fd = open("data.txt", O_RDWR);
 fd2 = creat("data2.txt", 0644); 
 if (fd2<0) {
    printf("\nFile open error\n");
    exit(1);
 }

 close(fd1);
 close(fd2);
} // end main
