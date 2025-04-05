#include <stdio.h>

#define MAX_BLOCKS 10
#define MAX_PROCESSES 10

int main() {
    int blockSizes[MAX_BLOCKS];
    int processSizes[MAX_PROCESSES];
    int allocation[MAX_PROCESSES];
    int m, n;

    // Initialize the allocation array
    for (int i = 0; i < MAX_PROCESSES; i++) {
        allocation[i] = -1;
    }

    // Input number of memory blocks
    printf("Enter the number of memory blocks (max %d): ", MAX_BLOCKS);
    if (scanf("%d", &m) != 1 || m <= 0 || m > MAX_BLOCKS) {
        printf("Invalid number of memory blocks!\n");
        return 1;
    }

    // Input block sizes
    printf("Enter the sizes of the %d memory blocks:\n", m);
    for (int i = 0; i < m; i++) {
        printf("Block %d: ", i + 1);
        if (scanf("%d", &blockSizes[i]) != 1) {
            printf("Invalid input for block size!\n");
            return 1;
        }
    }

    // Input number of processes
    printf("\nEnter the number of processes (max %d): ", MAX_PROCESSES);
    if (scanf("%d", &n) != 1 || n <= 0 || n > MAX_PROCESSES) {
        printf("Invalid number of processes!\n");
        return 1;
    }

    // Input process sizes
    printf("Enter the sizes of the %d processes:\n", n);
    for (int i = 0; i < n; i++) {
        printf("Process %d: ", i + 1);
        if (scanf("%d", &processSizes[i]) != 1) {
            printf("Invalid input for process size!\n");
            return 1;
        }
    }

    // Allocation logic (One process per block)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (blockSizes[j] >= processSizes[i]) {
                allocation[i] = j;
                // Mark this block as used
                blockSizes[j] = -1;
                break; // Move to the next process
            }
        }
    }

    // Print allocation results
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
