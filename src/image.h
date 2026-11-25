#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

typedef struct {
    int width;
    int height;
    int channels;

    uint8_t *data;

} Image;

Image load_image(const char *filename);

void save_image(
    const char *filename,
    const Image *image
);

void free_image(Image *image);

#endif
