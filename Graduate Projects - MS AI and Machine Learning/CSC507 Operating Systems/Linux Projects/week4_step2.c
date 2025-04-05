#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_FILE "file1.txt"
#define OUTPUT_FILE "newfile1.txt"

int main() {
    FILE *input_file, *output_file;
    int number;
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

    // Open the output file for writing
    output_file = fopen(OUTPUT_FILE, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        fclose(input_file);
        return EXIT_FAILURE;
    }

    // Process one row at a time
    while (fscanf(input_file, "%d", &number) != EOF) {
        number *= 2; // Double the number
        fprintf(output_file, "%d\n", number); // Write to output file
    }

    fclose(input_file);
    fclose(output_file);

    // End timing
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Output elapsed CPU time
    printf("Processing completed in %.2f seconds.\n", cpu_time_used);

    return 0;
}
