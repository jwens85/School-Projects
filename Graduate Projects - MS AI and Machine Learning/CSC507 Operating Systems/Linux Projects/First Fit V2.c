#include <stdio.h>
#define MAX_BLOCKS 10
#define MAX_PROCESSES 10

//TODO Step 1: Main function:
int main() {
    int blockSizes[MAX_BLOCKS];
    int processSizes[MAX_PROCESSES];
    int allocation[MAX_PROCESSES];
    int blockAllocated[MAX_BLOCKS]; // Array to track if a block is already allocated.
    int m, n;

    //TODO Step 2: Initialize the allocation Array:
    for (int i = 0; i < MAX_PROCESSES; i++) {
        allocation[i] = -1;
    }
    for (int i = 0; i < MAX_BLOCKS; i++) {
        blockAllocated[i] = 0; // New array initialized to 0, meaning no block is allocated yet.
    }

    //TODO Step 3: Input memory block sizes:
    printf("Enter the number of memory blocks (max %d): ", MAX_BLOCKS);
    scanf_s("%d", &m);
    if (m <= 0 || m > MAX_BLOCKS) {
        printf("Invalid number of memory blocks!\n");
        return 1;
    }
    printf("Enter the sizes of the %d memory blocks:\n", m);
    for (int i = 0; i < m; i++) {
        printf("Block %d: ", i + 1);
        scanf_s("%d", &blockSizes[i]);
    }

    //TODO Step 4: Input process sizes:
    printf("\nEnter the number of processes (max %d): ", MAX_PROCESSES);
    scanf_s("%d", &n);
    if (n <= 0 || n > MAX_PROCESSES) {
        printf("Invalid number of processes!\n");
        return 1;
    }
    printf("Enter the sizes of the %d processes:\n", n);
    for (int i = 0; i < n; i++) {
        printf("Process %d: ", i + 1);
        scanf_s("%d", &processSizes[i]);
    }

    //TODO Step 5: First-Fit Allocation Logic:
    for (int i = 0; i < n; i++) { // Loop through each process
        for (int j = 0; j < m; j++) { // Loop through each block
            if (blockSizes[j] >= processSizes[i] && blockAllocated[j] == 0) { // Check if block is unallocated and big enough
                allocation[i] = j; // Assign the block to the process
                blockAllocated[j] = 1; // Mark the block as allocated
                break; // Stop checking further blocks for this process
            }
        }
    }

    //TODO Step 6: Print allocation results
    printf("\nProcess No.\tProcess Size\tBlock No.\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t\t%d\t\t", i + 1, processSizes[i]);
        if (allocation[i] != -1) {
            printf("%d\n", allocation[i] + 1); // Print 1-based block number
        }
        else {
            printf("Not Allocated\n");
        }
    }

    return 0;
}
