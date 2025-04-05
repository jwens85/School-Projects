#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define NUM_LINES 1000000000 // 1 billion
#define NUM_THREADS 4        // Number of threads
#define CHUNK_SIZE 10000000  // 10 million numbers per chunk

// Thread function to generate random numbers and write to a temporary file
void *generate_random_numbers(void *arg) {
    long thread_id = (long)arg;
    char temp_filename[20];
    sprintf(temp_filename, "file_thread%ld.bin", thread_id);

    FILE *temp_file = fopen(temp_filename, "wb");
    if (temp_file == NULL) {
        perror("Error opening temporary file");
        pthread_exit(NULL);
    }

    int *buffer = malloc(CHUNK_SIZE * sizeof(int)); // Allocate buffer for random numbers
    if (buffer == NULL) {
        perror("Failed to allocate buffer");
        fclose(temp_file);
        pthread_exit(NULL);
    }

    srand(time(0) + thread_id); // Unique seed for each thread

    long numbers_to_generate = NUM_LINES / NUM_THREADS;
    for (long i = 0; i < numbers_to_generate; i += CHUNK_SIZE) {
        long chunk_size = (i + CHUNK_SIZE <= numbers_to_generate) ? CHUNK_SIZE : (numbers_to_generate - i);

        // Generate a chunk of random numbers
        for (long j = 0; j < chunk_size; j++) {
            buffer[j] = rand();
        }

        // Write the chunk to the temporary file
        fwrite(buffer, sizeof(int), chunk_size, temp_file);
    }

    free(buffer);
    fclose(temp_file);
    pthread_exit(NULL);
}

int main() {
    time_t start_time, end_time;
    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    // Get and display the start wall-clock timestamp
    time(&start_time);
    printf("Start time (wall-clock): %s", ctime(&start_time));

    // Start the CPU timer
    cpu_start = clock();

    // Create and launch threads
    pthread_t threads[NUM_THREADS];
    for (long i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, generate_random_numbers, (void *)i);
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Merge temporary files into the final output file
    FILE *output_file = fopen("file2.bin", "wb");
    if (output_file == NULL) {
        perror("Error opening output file");
        return 1;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        char temp_filename[20];
        sprintf(temp_filename, "file_thread%d.bin", i);

        FILE *temp_file = fopen(temp_filename, "rb");
        if (temp_file == NULL) {
            perror("Error opening temporary file");
            fclose(output_file);
            return 1;
        }

        int *buffer = malloc(CHUNK_SIZE * sizeof(int));
        if (buffer == NULL) {
            perror("Failed to allocate buffer");
            fclose(temp_file);
            fclose(output_file);
            return 1;
        }

        // Read and write from the temporary file in chunks
        size_t read_count;
        while ((read_count = fread(buffer, sizeof(int), CHUNK_SIZE, temp_file)) > 0) {
            fwrite(buffer, sizeof(int), read_count, output_file);
        }

        free(buffer);
        fclose(temp_file);
        remove(temp_filename); // Delete temporary file after merging
    }

    fclose(output_file);

    // Stop the CPU timer
    cpu_end = clock();
    cpu_time_used = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

    // Get and display the end wall-clock timestamp
    time(&end_time);
    printf("End time (wall-clock): %s", ctime(&end_time));

    // Calculate the elapsed wall-clock time
    double elapsed_seconds = difftime(end_time, start_time);
    int elapsed_minutes = (int)elapsed_seconds / 60;
    int elapsed_remaining_seconds = (int)elapsed_seconds % 60;

    // Display the elapsed wall-clock time
    printf("Elapsed time (wall-clock): %d minutes, %d seconds\n", elapsed_minutes, elapsed_remaining_seconds);

    // Display the CPU time in minutes and seconds
    int cpu_minutes = (int)cpu_time_used / 60;
    int cpu_remaining_seconds = (int)cpu_time_used % 60;
    printf("Elapsed time (CPU): %d minutes, %d seconds (%.2f seconds total)\n", cpu_minutes, cpu_remaining_seconds, cpu_time_used);

    return 0;
}
