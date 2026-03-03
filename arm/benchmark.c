#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    printf("QuantEdge ARM Binary Running!\n");
    printf("CPU: armeabi-v7a\n");
    printf("Android API: 16\n");

    clock_t start = clock();
    volatile long sum = 0;
    int i;
    for(i = 0; i < 10000000; i++) {
        sum += i;
    }
    clock_t end = clock();

    double ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
    printf("10M iterations: %.2f ms\n", ms);
    printf("Sum: %ld\n", sum);
    printf("Binary works on Samsung GT-S7392!\n");
    return 0;
}
