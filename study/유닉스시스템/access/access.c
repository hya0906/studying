#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
char *filename = "test.txt";

if(access(filename, R_OK)==-1){
 printf("\nUser cannot read this file %s\n", filename);
 exit(1);
}//end if()

printf("\n%s readable, proceeding\n",filename);
//exit(0) good end
}//end main()
