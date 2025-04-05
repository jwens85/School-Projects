#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    FILE *file = fopen("file2.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    srand(time(0));

    for (int i = 0; i < 1000; i++) {
        fprintf(file, "%d\n", rand());
    }

    fclose(file);

    printf("File file2.txt has been created with 1,000 random numbers.\n");
    return 0;
}

// OpenAI. (2024). Grimoire: AI Chat Retrieved November 24, 2024, from https://chat.openai.com
