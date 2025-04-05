#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define NUM_LINES 1000000
#define NUM_THREADS 4
#define CHUNK_SIZE 10000000
#define BUFFER_SIZE 1000000

typedef struct {
    int thread_id;
    FILE *output_file;
    pthread_mutex_t *file_mutex;
} merge_data_t;

void *generate_random_numbers(void *arg) {
    long thread_id = (long)arg;
    char temp_filename[20];
    sprintf(temp_filename, "file_thread%ld.bin", thread_id);

    FILE *temp_file = fopen(temp_filename, "wb");
    if (temp_file == NULL) {
        perror("Error opening temporary file");
        pthread_exit(NULL);
    }

    int *buffer = malloc(CHUNK_SIZE * sizeof(int));
    if (buffer == NULL) {
        perror("Failed to allocate buffer");
        fclose(temp_file);
        pthread_exit(NULL);
    }

    srand(time(0) + thread_id);
    long numbers_to_generate = NUM_LINES / NUM_THREADS;
    for (long i = 0; i < numbers_to_generate; i += CHUNK_SIZE) {
        long chunk_size = (i + CHUNK_SIZE <= numbers_to_generate) ? CHUNK_SIZE : (numbers_to_generate - i);
        for (long j = 0; j < chunk_size; j++) {
            buffer[j] = rand();
        }
        fwrite(buffer, sizeof(int), chunk_size, temp_file);
    }

    free(buffer);
    fclose(temp_file);
    pthread_exit(NULL);
}

void *merge_to_text(void *arg) {
    merge_data_t *data = (merge_data_t *)arg;
    int thread_id = data->thread_id;

    char temp_filename[20];
    sprintf(temp_filename, "file_thread%d.bin", thread_id);

    FILE *temp_file = fopen(temp_filename, "rb");
    if (temp_file == NULL) {
        perror("Error opening temporary file");
        pthread_exit(NULL);
    }

    int *binary_buffer = malloc(CHUNK_SIZE * sizeof(int));
    if (binary_buffer == NULL) {
        perror("Failed to allocate binary buffer");
        fclose(temp_file);
        pthread_exit(NULL);
    }

    char *text_buffer = malloc(BUFFER_SIZE * sizeof(char));
    if (text_buffer == NULL) {
        perror("Failed to allocate text buffer");
        fclose(temp_file);
        free(binary_buffer);
        pthread_exit(NULL);
    }

    size_t read_count;
    while ((read_count = fread(binary_buffer, sizeof(int), CHUNK_SIZE, temp_file)) > 0) {
        size_t text_index = 0;
        for (size_t i = 0; i < read_count; i++) {
            text_index += snprintf(&text_buffer[text_index], BUFFER_SIZE - text_index, "%d\n", binary_buffer[i]);
            if (text_index >= BUFFER_SIZE - 50) {
                pthread_mutex_lock(data->file_mutex);
                fwrite(text_buffer, sizeof(char), text_index, data->output_file);
                pthread_mutex_unlock(data->file_mutex);
                text_index = 0;
            }
        }
        if (text_index > 0) {
            pthread_mutex_lock(data->file_mutex);
            fwrite(text_buffer, sizeof(char), text_index, data->output_file);
            pthread_mutex_unlock(data->file_mutex);
        }
    }

    free(binary_buffer);
    free(text_buffer);
    fclose(temp_file);

        remove(temp_filename);
    pthread_exit(NULL);
}

int main() {
    time_t start_time, end_time;
    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    time(&start_time);
    printf("Start time (wall-clock): %s", ctime(&start_time));

    cpu_start = clock();

    pthread_t generator_threads[NUM_THREADS];
    for (long i = 0; i < NUM_THREADS; i++) {
        pthread_create(&generator_threads[i], NULL, generate_random_numbers, (void *)i);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(generator_threads[i], NULL);
    }

    FILE *output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        return 1;
    }

    pthread_t merger_threads[NUM_THREADS];
    pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;
    merge_data_t merge_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        merge_data[i].thread_id = i;
        merge_data[i].output_file = output_file;
        merge_data[i].file_mutex = &file_mutex;
        pthread_create(&merger_threads[i], NULL, merge_to_text, &merge_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(merger_threads[i], NULL);
    }

    fclose(output_file);

    cpu_end = clock();
    cpu_time_used = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

    time(&end_time);
    printf("End time (wall-clock): %s", ctime(&end_time));

    double elapsed_seconds = difftime(end_time, start_time);
    int elapsed_minutes = (int)elapsed_seconds / 60;
    int elapsed_remaining_seconds = (int)elapsed_seconds % 60;
    printf("Elapsed time (wall-clock): %d minutes, %d seconds\n", elapsed_minutes, elapsed_remaining_seconds);

    int cpu_minutes = (int)cpu_time_used / 60;
    int cpu_remaining_seconds = (int)cpu_time_used % 60;
    printf("Elapsed time (CPU): %d minutes, %d seconds (%.2f seconds total)\n", cpu_minutes, cpu_remaining_seconds, cpu_time_used);

    return 0;
}
