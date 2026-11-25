#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int read_token(FILE *file, char *token)
{
    int c;
    int i = 0;

    while ((c = fgetc(file)) != EOF) {

        if (c == '#') {
            while ((c = fgetc(file)) != EOF && c != '\n');
            continue;
        }

        if (!isspace((unsigned char)c))
            break;
    }

    if (c == EOF)
        return 0;

    do {
        if (i < 63)
            token[i++] = (char)c;

        c = fgetc(file);

    } while (c != EOF && !isspace((unsigned char)c));

    token[i] = '\0';

    return 1;
}

Image load_image(const char *filename)
{
    Image image = {0, 0, 3, NULL};

    FILE *file = fopen(filename, "rb");

    if (!file) {
        fprintf(stderr, "Cannot open image: %s\n", filename);
        return image;
    }

    char token[64];

    if (!read_token(file, token) || strcmp(token, "P6") != 0) {
        fprintf(stderr, "Unsupported image format.\n");
        fclose(file);
        return image;
    }

    if (!read_token(file, token)) {
        fclose(file);
        return image;
    }

    image.width = atoi(token);

    if (!read_token(file, token)) {
        fclose(file);
        return image;
    }

    image.height = atoi(token);

    if (!read_token(file, token)) {
        fclose(file);
        return image;
    }

    int max_value = atoi(token);

    if (image.width <= 0 || image.height <= 0 ||
        max_value != 255) {

        fprintf(stderr, "Invalid PPM dimensions or max value.\n");
        fclose(file);
        return (Image){0, 0, 3, NULL};
    }

    size_t size =
        (size_t)image.width * image.height * image.channels;

    image.data = malloc(size);

    if (!image.data) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        return (Image){0, 0, 3, NULL};
    }

    if (fread(image.data, 1, size, file) != size) {

        fprintf(stderr, "Error reading image pixels.\n");

        free(image.data);
        image.data = NULL;

        fclose(file);

        return image;
    }

    fclose(file);

    return image;
}

void save_image(const char *filename, const Image *image)
{
    FILE *file = fopen(filename, "wb");

    if (!file) {
        fprintf(stderr, "Cannot save image.\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n",
            image->width, image->height);

    size_t size =
        (size_t)image->width *
        image->height *
        image->channels;

    fwrite(image->data, 1, size, file);

    fclose(file);
}

void free_image(Image *image)
{
    free(image->data);

    image->data = NULL;
    image->width = 0;
    image->height = 0;
}
