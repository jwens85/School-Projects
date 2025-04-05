#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#define MAX_BLOCKS 10
#define MAX_PROCESSES 10

int main() {
    int blockSizes[MAX_BLOCKS];
    int processSizes[MAX_PROCESSES];
    int allocation[MAX_PROCESSES];
    int blockAllocated[MAX_BLOCKS];
    int m, n;

    for (int i = 0; i < MAX_PROCESSES; i++) {
        allocation[i] = -1;
    }
    for (int i = 0; i < MAX_BLOCKS; i++) {
        blockAllocated[i] = 0;
    }

    printf("Enter the number of memory blocks (max %d): ", MAX_BLOCKS);
    scanf("%d", &m);
    if (m <= 0 || m > MAX_BLOCKS) {
        printf("Invalid number of memory blocks!\n");
        return 1;
    }
    printf("Enter the sizes of the %d memory blocks:\n", m);
    for (int i = 0; i < m; i++) {
        printf("Block %d: ", i + 1);
        scanf("%d", &blockSizes[i]);
    }

    printf("\nEnter the number of processes (max %d): ", MAX_PROCESSES);
    scanf("%d", &n);
    if (n <= 0 || n > MAX_PROCESSES) {
        printf("Invalid number of processes!\n");
        return 1;
    }
    printf("Enter the sizes of the %d processes:\n", n);
    for (int i = 0; i < n; i++) {
        printf("Process %d: ", i + 1);
        scanf("%d", &processSizes[i]);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (blockSizes[j] >= processSizes[i] && blockAllocated[j] == 0) {
                allocation[i] = j;
                blockAllocated[j] = 1;
                break;
            }
        }
    }

    printf("\nProcess No.\tProcess Size\tBlock No.\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t\t%d\t\t", i + 1, processSizes[i]);
        if (allocation[i] != -1) {
            printf("%d\n", allocation[i] + 1);
        }
        else {
            printf("Not Allocated\n");
        }
    }

    return 0;
}
