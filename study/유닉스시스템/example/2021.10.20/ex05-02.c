#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

int main()
{
 mkdir("test_dir1", 0755);
 mkdir("test_dir2", 0755);
 
 rmdir("test_dir2");
}
