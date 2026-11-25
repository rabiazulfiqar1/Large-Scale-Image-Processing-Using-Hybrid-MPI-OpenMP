#include "gaussian.h"

#include <mpi.h>
#include <omp.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static const float kernel[3][3] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}
};

// Calculate pixel using local image plus halo rows
static unsigned char distributed_pixel(
    const unsigned char *data,
    int width,
    int total_rows,
    int y,
    int x,
    int channel
) {
    float sum = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int row = y + ky;
            int col = x + kx;

            // Clamp horizontal boundaries
            if (col < 0) col = 0;
            if (col >= width) col = width - 1;

            // Vertical clamping is handled by
            // physical halo initialization.

            int index =
                (row * width + col) * 3 + channel;

            sum += data[index] *
                   kernel[ky + 1][kx + 1];
        }
    }

    int result = (int)roundf(sum / 16.0f);

    if (result < 0) result = 0;
    if (result > 255) result = 255;

    return (unsigned char)result;
}

void gaussian_distributed(
    const Image *input,
    Image *output,
    int use_openmp
) {
    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = 0;
    int height = 0;

    if (rank == 0) {
        width = input->width;
        height = input->height;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide rows among processes
    int base = height / processes;
    int remainder = height % processes;

    int local_rows = base + (rank < remainder);

    if (local_rows < 1) {
        if (rank == 0)
            fprintf(stderr, "Too many processes for image height.\n");

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *counts = malloc(processes * sizeof(int));
    int *displacements = malloc(processes * sizeof(int));

    if (!counts || !displacements)
        MPI_Abort(MPI_COMM_WORLD, 1);

    int offset = 0;

    for (int p = 0; p < processes; p++) {

        int rows = base + (p < remainder);

        counts[p] = rows * width * 3;
        displacements[p] = offset;

        offset += counts[p];
    }

    int local_size = local_rows * width * 3;
    int row_size = width * 3;

    unsigned char *local_data =
        malloc((size_t)local_size);

    unsigned char *local_output =
        malloc((size_t)local_size);

    if (!local_data || !local_output)
        MPI_Abort(MPI_COMM_WORLD, 1);

    // Distribute image
    MPI_Scatterv(
        rank == 0 ? input->data : NULL,
        counts,
        displacements,
        MPI_UNSIGNED_CHAR,

        local_data,
        local_size,
        MPI_UNSIGNED_CHAR,

        0,
        MPI_COMM_WORLD
    );

    // Allocate extra top and bottom halo rows
    int extended_rows = local_rows + 2;

    unsigned char *extended =
        calloc((size_t)extended_rows * row_size, 1);

    if (!extended)
        MPI_Abort(MPI_COMM_WORLD, 1);

    // Copy owned rows into middle
    memcpy(
        extended + row_size,
        local_data,
        (size_t)local_size
    );

    int previous = (rank == 0)
        ? MPI_PROC_NULL : rank - 1;

    int next = (rank == processes - 1)
        ? MPI_PROC_NULL : rank + 1;

    // Top physical boundary
    if (previous == MPI_PROC_NULL) {
        memcpy(
            extended,
            local_data,
            row_size
        );
    }

    // Bottom physical boundary
    if (next == MPI_PROC_NULL) {
        memcpy(
            extended + (size_t)(local_rows + 1) * row_size,
            local_data + (size_t)(local_rows - 1) * row_size,
            row_size
        );
    }

    // Exchange halo rows
    MPI_Sendrecv(
        local_data,
        row_size,
        MPI_UNSIGNED_CHAR,
        previous,
        0,

        extended + (size_t)(local_rows + 1) * row_size,
        row_size,
        MPI_UNSIGNED_CHAR,
        next,
        0,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        local_data + (size_t)(local_rows - 1) * row_size,
        row_size,
        MPI_UNSIGNED_CHAR,
        next,
        1,

        extended,
        row_size,
        MPI_UNSIGNED_CHAR,
        previous,
        1,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    int y;
    // Process assigned rows
    #pragma omp parallel for schedule(static) if(use_openmp)
    for (y = 0; y < local_rows; y++) {
        int x;
        for (x = 0; x < width; x++) {
            int c;
            for (c = 0; c < 3; c++) {

                int index = (y * width + x) * 3 + c;

                local_output[index] =
                    distributed_pixel(
                        extended,
                        width,
                        extended_rows,
                        y + 1,
                        x,
                        c
                    );
            }
        }
    }

    // Prepare output on root
    if (rank == 0) {
        output->width = width;
        output->height = height;
        output->channels = 3;

        output->data =
            malloc((size_t)width * height * 3);

        if (!output->data)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Gather processed rows
    MPI_Gatherv(
        local_output,
        local_size,
        MPI_UNSIGNED_CHAR,

        rank == 0 ? output->data : NULL,
        counts,
        displacements,
        MPI_UNSIGNED_CHAR,

        0,
        MPI_COMM_WORLD
    );

    free(local_data);
    free(local_output);
    free(extended);
    free(counts);
    free(displacements);
}
