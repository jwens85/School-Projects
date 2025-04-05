#include <stdio.h>    // Standard I/O library for file handling and printing
#include <stdlib.h>   // Standard library for memory allocation, random number generation
#include <time.h>     // Library for time-related functions (used for seeding random numbers)
#include <pthread.h>  // POSIX thread library for multithreading

// Define constants
#define NUM_LINES 1000000000 // Total number of random numbers to generate (1 billion)
#define NUM_THREADS 4        // Number of threads to use
#define CHUNK_SIZE 10000000  // Number of random numbers each thread handles in a chunk

// Function executed by each thread to generate random numbers and write them to a temporary file
void *generate_random_numbers(void *arg) {
    long thread_id = (long)arg; // Get the thread ID passed as an argument

    // Create a unique temporary file name for each thread
    char temp_filename[20];
    sprintf(temp_filename, "file_thread%ld.bin", thread_id); // e.g., file_thread0.bin, file_thread1.bin

    // Open the temporary file in binary write mode
    FILE *temp_file = fopen(temp_filename, "wb");
    if (temp_file == NULL) { // Check for file opening errors
        perror("Error opening temporary file");
        pthread_exit(NULL); // Exit the thread if file opening fails
    }

    // Allocate memory for a buffer to hold CHUNK_SIZE integers
    int *buffer = malloc(CHUNK_SIZE * sizeof(int)); 
    if (buffer == NULL) { // Check for memory allocation errors
        perror("Failed to allocate buffer");
        fclose(temp_file);
        pthread_exit(NULL);
    }

    // Seed the random number generator uniquely for each thread
    srand(time(0) + thread_id);

    // Calculate the number of random numbers this thread needs to generate
    long numbers_to_generate = NUM_LINES / NUM_THREADS;
    for (long i = 0; i < numbers_to_generate; i += CHUNK_SIZE) {
        // Determine the size of the current chunk
        long chunk_size = (i + CHUNK_SIZE <= numbers_to_generate) ? CHUNK_SIZE : (numbers_to_generate - i);

        // Generate random numbers for the current chunk
        for (long j = 0; j < chunk_size; j++) {
            buffer[j] = rand();
        }

        // Write the generated random numbers to the temporary file
        fwrite(buffer, sizeof(int), chunk_size, temp_file);
    }

    // Free the allocated memory and close the temporary file
    free(buffer);
    fclose(temp_file);

    // Exit the thread
    pthread_exit(NULL);
}

int main() {
    time_t start_time, end_time; // Variables to store start and end wall-clock time
    clock_t cpu_start, cpu_end; // Variables to store start and end CPU time
    double cpu_time_used;       // Variable to calculate total CPU time used

    // Record and display the start wall-clock time
    time(&start_time);
    printf("Start time (wall-clock): %s", ctime(&start_time));

    // Start the CPU timer
    cpu_start = clock();

    // Create an array of threads
    pthread_t threads[NUM_THREADS];
    for (long i = 0; i < NUM_THREADS; i++) {
        // Create and launch a thread for each segment of the workload
        pthread_create(&threads[i], NULL, generate_random_numbers, (void *)i);
    }

    // Wait for all threads to complete their tasks
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Open the final output file to merge all temporary files
    FILE *output_file = fopen("file2.bin", "wb");
    if (output_file == NULL) { // Check for file opening errors
        perror("Error opening output file");
        return 1; // Exit with error
    }

    // Loop through all temporary files created by the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        char temp_filename[20];
        sprintf(temp_filename, "file_thread%d.bin", i); // Name of the temporary file

        // Open the temporary file in binary read mode
        FILE *temp_file = fopen(temp_filename, "rb");
        if (temp_file == NULL) { // Check for file opening errors
            perror("Error opening temporary file");
            fclose(output_file);
            return 1; // Exit with error
        }

        // Allocate a buffer to read the data in chunks
        int *buffer = malloc(CHUNK_SIZE * sizeof(int));
        if (buffer == NULL) { // Check for memory allocation errors
            perror("Failed to allocate buffer");
            fclose(temp_file);
            fclose(output_file);
            return 1; // Exit with error
        }

        // Read data from the temporary file and write it to the final output file
        size_t read_count;
        while ((read_count = fread(buffer, sizeof(int), CHUNK_SIZE, temp_file)) > 0) {
            fwrite(buffer, sizeof(int), read_count, output_file);
        }

        // Free the allocated memory and close the temporary file
        free(buffer);
        fclose(temp_file);

        // Delete the temporary file after merging its data
        remove(temp_filename);
    }

    // Close the final output file
    fclose(output_file);

    // Stop the CPU timer
    cpu_end = clock();
    cpu_time_used = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

    // Record and display the end wall-clock time
    time(&end_time);
    printf("End time (wall-clock): %s", ctime(&end_time));

    // Calculate and display the elapsed wall-clock time
    double elapsed_seconds = difftime(end_time, start_time);
    int elapsed_minutes = (int)elapsed_seconds / 60;
    int elapsed_remaining_seconds = (int)elapsed_seconds % 60;
    printf("Elapsed time (wall-clock): %d minutes, %d seconds\n", elapsed_minutes, elapsed_remaining_seconds);

    // Calculate and display the elapsed CPU time
    int cpu_minutes = (int)cpu_time_used / 60;
    int cpu_remaining_seconds = (int)cpu_time_used % 60;
    printf("Elapsed time (CPU): %d minutes, %d seconds (%.2f seconds total)\n", cpu_minutes, cpu_remaining_seconds, cpu_time_used);

    return 0; // Exit the program successfully
}
