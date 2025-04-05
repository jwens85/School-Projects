//The # symbol in C indicates a preprocessing directive that will run before compilation.
#include <stdio.h>//Include the standard I/O library so we can use printf and scanf_s commands.
#define MAX_BLOCKS 10//Preprocessor macro to replace every instance of MAX_BLOCKS with the value 10 before compilation.
#define MAX_PROCESSES 10//Preprocessor macro to replace every instance of MAX_PROCESSES with the value 10 before compilation.

/*
-C is different from Python here with main() being the entry point of a program instead of def main()
	to indicate a user defined function.
-This is mandatory for all C programs. Arguments would be placed within the () if needed, and {} encloses the
	body of the function. The OS calls main() automatically here and it doesn't need to be called explicitly in the code.
-Here we are using int main() to return an integer to indicate the program's success or failure to the OS.
	return0; would indicate that the program ran successfully.
	return 1; would indicate that there was an execution issue.
-C seems to be a much lower level language, operating closer to the bare metal, needing to be explicit
	about almost everything. Variable types will need to be declared (int, float, etc), and memory will need to be
	managed manually with *ptr, malloc, and free. Syntax differences include the need for ; and {}.
-This makes the C language extremely fast compared to interpreted languages that have a lot of abstraction.
-Camel case, while not required in C is a popular convention for readability and consistency.
	Constants and macros (MAX_BLOCKS) are written in upper case to indicate immutability.
	Variables and function names (blockSizes) are written in camel case to indicate that values may change in execution.
*/

//TODO Step 1: Main function:
int main() {
	int blockSizes[MAX_BLOCKS]; //Array to store sizes of memory blocks.
	int processSizes[MAX_PROCESSES]; //Array to store the size of processes.
	int allocation[MAX_PROCESSES];//Array to track which block each process is assigned to.
	int m, n; //Number of blocks (m) and processes (n). Multiple variables can be declared in the same statement in C.

//TODO Step 2: Initialize the allocation Array:
			// A for loop in C has four main parts for(initialization; condition; increment){loop body}.
		//Initialize the allocation array to -1 for each element up to the length of MAX_PROCESSES.
	for (int i = 0; i < MAX_PROCESSES; i++) {
		allocation[i] = -1; // -1 here means that a process is not allocated.
	}

//TODO Step 3: Input memory block sizes:
	printf("Enter the number of memory blocks (max %d): ", MAX_BLOCKS);//%d is a format specifier for an integer value.
	scanf_s("%d", &m); //scanf_s() is part of stdio.h and used to scan input provided to the user and store it in the specified variable m.
	//We have to use scanf_s instead of plain scanf because visual studio considers scanf unsafe and gives errors at compilation time. 
	//Validation check to ensure the number of memory blocks is within the acceptable range.
	if (m <= 0 || m > MAX_BLOCKS) { //If m is <= 0 || (logical OR) m is > MAX_BLOCKS (either condition will execute the if block).
		printf("Invalid number of memory blocks!\n"); //Error message will print to inform the user about the mistake.
		return 1;  // Exit the program.
	}
	
	printf("Enter the sizes of the %d memory blocks:\n", m);//Message to ask the user to enter the size of the memory blocks and start a new line.
	for (int i = 0; i < m; i++) {//For loop that iterates m times.
		printf("Block %d: ", i + 1);//Prompts user to enter the size of a specific block. The %d is replaced by i + 1 to show a 1-based block number.
		scanf_s("%d", &blockSizes[i]);//Reads the size of the current block user input and store the value in the ith element in the blockSizes array. 
	}
	
//TODO Step 4: Input process sizes:
	printf("\nEnter the number of processes (max %d): ", MAX_PROCESSES);//Prompt the user to input the number of processes starting on a new line. 
	scanf_s("%d", &n);//Read an integer input from the user and store it in the variable n.
	if (n <= 0 || n > MAX_PROCESSES) {//If n <=0 || (logical OR) n > MAX PROCESSES (either condition will execute the if block).
		printf("Invalid number of processes!\n");//Error message will inform the user about the mistake. 
		return 1;  // Exit the program.
	}
	printf("Enter the sizes of the %d processes:\n", n);//Prompt the user to input the size of the processes and start a new line. 
	for (int i = 0; i < n; i++) {//For loop that iterates n times.
		printf("Process %d: ", i + 1);//Prompts the user to enter the size of a specific process. The %d is replaced by i + 1 to show a 1-based process number.
		scanf_s("%d", &processSizes[i]);//Reads the size of the current process user input and store the value in the ith element in the processSizes array. 
	}
//TODO Step 5: First-Fit Allocation Logic:
	for (int i = 0; i < n; i++) {//Outer for loop: Iterates through each process in the processSizes array, which is referenced in the inner loop.
		for (int j = 0; j < m; j++) {//Inner for loop: Iterates through each memory block for the current process (processSizes[i]). Time Complexity is O(n*m).
			if (blockSizes[j] >= processSizes[i]) {//Check if the current block (blockSizes[j]) can fit the current process (processSizes[i]).
				allocation[i] = j;//If the block size is big enough, assign the current block j to process index i.
				blockSizes[j] -= processSizes[i];//The size of block [j] will then be reduced by processSize [i] leading to some external fragmentation. 
				break;//Exit the inner loop and start the next outer loop iteration if i is still less than n.
			}	//If we were implementing a best-fit allocation, the logic would continue to run through all blcoks to find the smallest block to fit the process. 
		}
	}
//TODO Step 6: Print allocation results
	printf("\nProcess No.\tProcess Size\tBlock No.\n");//Print a table header to display the process allocation results \t makes a horizontal tab for alignment.
	for (int i = 0; i < n; i++) {//Loop through each process to display its size and the block it was allocated to.
		printf("%d\t\t%d\t\t", i + 1, processSizes[i]);// Print process number and size with tab spacing for alignment. Note the 2 instances of %d placeholders.
		if (allocation[i] != -1) {//If the current process i was assigned to a block (remember we initialized allocation array to be -1 to represent empty index).
			printf("%d\n", allocation[i] + 1);//Print the 1-based index of the block allocated to the current process and start a new line.
		}
		else {//If the current process has not been allocated to any block.
			printf("Not Allocated\n");//Print that it has not been allocated and start a new line.
		}
	}

	return 0;//Program ran successfully.
}