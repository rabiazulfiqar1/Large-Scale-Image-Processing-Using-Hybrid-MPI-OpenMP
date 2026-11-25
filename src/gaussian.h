#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "image.h"

void gaussian_sequential(
    const Image *input,
    Image *output
);

void gaussian_openmp(
    const Image *input,
    Image *output
);

void gaussian_distributed(
    const Image *input,
    Image *output,
    int use_openmp
);

#endif
