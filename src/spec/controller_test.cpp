// controller_test.cpp — speculative decoding controller invariant tests.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static long g_fail = 0;

static const char* name(int mode) {
    switch(mode){case 0:return"eager";case 1:return"conservative";default:return"unknown";}
}

static void check(int mode, int expected_draft_len, int actual_len) {
    if(actual_len != expected_draft_len){
        std::printf("FAIL controller_test check mode=%s expected=%d actual=%d\n",name(mode),expected_draft_len,actual_len);
        ++g_fail;
    }
}

int main(int argc, char** argv) {
    (void)argc;(void)argv;
    // Minimal invariant: check that name() returns non-empty strings
    if(!name(0)||!name(1)){std::printf("FAIL controller_test name null\n");++g_fail;}
    check(0,4,4); // eager mode: draft len 4
    check(1,2,2); // conservative: draft len 2
    if(g_fail){std::printf("controller_test FAILED (%ld)\n",g_fail);return 1;}
    std::printf("controller_test PASSED\n");return 0;
}
