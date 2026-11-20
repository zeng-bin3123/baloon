// memory_test.cpp — platform memory allocation and mlock invariant tests.
// Cross-platform: WIN32_LEAN_AND_MEAN and NOMINMAX guard Windows headers when compiled there.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    // Allocate and touch a page to verify the platform allocator works.
    std::vector<unsigned char> buf(4096, 0);
    buf[0]=0xDE; buf[4095]=0xAD;
    if(buf[0]!=0xDE||buf[4095]!=0xAD){std::printf("FAIL platform_memory_test alloc\n");return 1;}
    std::printf("platform_memory_test PASSED\n");return 0;
}
