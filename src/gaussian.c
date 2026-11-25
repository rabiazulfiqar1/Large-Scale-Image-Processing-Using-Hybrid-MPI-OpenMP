#include "gaussian.h"

#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// 3x3 Gaussian kernel
static const float kernel[3][3] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}
};

// Calculate one output pixel channel
static unsigned char calculate_pixel(
    const unsigned char *input,
    int width,
    int height,
    int y,
    int x,
    int channel
) {
    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {

        for (int kx = -1; kx <= 1; kx++) {

            int row = y + ky;
            int col = x + kx;

            // Clamp boundaries
            if (row < 0) row = 0;
            if (row >= height) row = height - 1;

            if (col < 0) col = 0;
            if (col >= width) col = width - 1;

            int index =
                (row * width + col) * 3 + channel;

            sum += input[index] *
                   kernel[ky + 1][kx + 1];
        }
    }

    sum /= 16.0f;

    // Round and clamp
    int result = (int)roundf(sum);

    if (result < 0) result = 0;
    if (result > 255) result = 255;

    return (unsigned char)result;
}

// Sequential implementation
void gaussian_sequential(
    const Image *input,
    Image *output
) {
    output->width = input->width;
    output->height = input->height;
    output->channels = 3;

    size_t size =
        (size_t)input->width *
        input->height * 3;

    output->data = malloc(size);

    if (!output->data)
        return;

    for (int y = 0; y < input->height; y++) {

        for (int x = 0; x < input->width; x++) {

            for (int c = 0; c < 3; c++) {

                int index = (y * input->width + x) * 3 + c;

                output->data[index] =
                    calculate_pixel(
                        input->data,
                        input->width,
                        input->height,
                        y, x, c
                    );
            }
        }
    }
}

// OpenMP implementation
void gaussian_openmp(
    const Image *input,
    Image *output
) {
    output->width = input->width;
    output->height = input->height;
    output->channels = 3;

    size_t size =
        (size_t)input->width *
        input->height * 3;

    output->data = malloc(size);

    if (!output->data)
        return;

    int y;
    #pragma omp parallel for schedule(static)
    for (y = 0; y < input->height; y++) {
        int x;
        for (x = 0; x < input->width; x++) {
            int c;
            for (c = 0; c < 3; c++) {

                int index = (y * input->width + x) * 3 + c;

                output->data[index] =
                    calculate_pixel(
                        input->data,
                        input->width,
                        input->height,
                        y, x, c
                    );
            }
        }
    }
}
