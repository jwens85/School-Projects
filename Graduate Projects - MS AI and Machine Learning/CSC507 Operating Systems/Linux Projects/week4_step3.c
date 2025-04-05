#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_FILE "file1.txt"
#define OUTPUT_FILE "newfile1.txt"

void process_part(FILE *input_file, FILE *output_file, long start_line, long end_line) {
    fseek(input_file, 0, SEEK_SET); // Reset file pointer to the start
    char buffer[32];               // Buffer to hold a single line
    long current_line = 0;

    // Skip lines until the starting line
    while (current_line < start_line && fgets(buffer, sizeof(buffer), input_file) != NULL) {
        current_line++;
    }

    // Read and process lines in the specified range
    for (; current_line < end_line && fgets(buffer, sizeof(buffer), input_file) != NULL; current_line++) {
        int number = atoi(buffer);
        number *= 2; // Double the number
        fprintf(output_file, "%d\n", number);
    }
}

int main() {
    FILE *input_file, *output_file;
    long total_lines = 0;
    char temp;
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

    // Count the total number of lines in the file
    while ((temp = fgetc(input_file)) != EOF) {
        if (temp == '\n') {
            total_lines++;
        }
    }

    // Define the split points
    long midpoint = total_lines / 2;

    // Open the output file for writing
    output_file = fopen(OUTPUT_FILE, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        fclose(input_file);
        return EXIT_FAILURE;
    }

    // Process each part separately
    process_part(input_file, output_file, 0, midpoint); // First half
    process_part(input_file, output_file, midpoint, total_lines); // Second half

    fclose(input_file);
    fclose(output_file);

    // End timing
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Output elapsed CPU time
    printf("Processing completed in %.2f seconds.\n", cpu_time_used);

    return 0;
}
