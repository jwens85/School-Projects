#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_FILE "file1.txt"
#define OUTPUT_FILE "newfile1.txt"

int main() {
    FILE *input_file, *output_file;
    int *numbers;
    long num_count = 0;
    clock_t start, end;
    double cpu_time_used;

    // Start timing
    start = clock();

    // Open the input file for reading
    input_file = fopen(INPUT_FILE, "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    // Count the number of lines in the file
    char temp;
    while ((temp = fgetc(input_file)) != EOF) {
        if (temp == '\n') {
            num_count++;
        }
    }
    rewind(input_file); // Reset file pointer to the beginning of the file

    // Allocate memory to hold all numbers
    numbers = (int *)malloc(num_count * sizeof(int));
    if (numbers == NULL) {
        perror("Error allocating memory");
        fclose(input_file);
        return EXIT_FAILURE;
    }

    // Read all numbers into memory
    for (long i = 0; i < num_count; i++) {
        if (fscanf(input_file, "%d", &numbers[i]) != 1) {
            perror("Error reading number from file");
            free(numbers);
            fclose(input_file);
            return EXIT_FAILURE;
        }
    }
    fclose(input_file);

    // Process numbers: double each value
    for (long i = 0; i < num_count; i++) {
        numbers[i] *= 2;
    }

    // Write processed numbers to the output file
    output_file = fopen(OUTPUT_FILE, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        free(numbers);
        return EXIT_FAILURE;
    }

    for (long i = 0; i < num_count; i++) {
        fprintf(output_file, "%d\n", numbers[i]);
    }
    fclose(output_file);

    // Free allocated memory
    free(numbers);

    // End timing
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Output elapsed CPU time
    printf("Processing completed in %.2f seconds.\n", cpu_time_used);

    return 0;
}
