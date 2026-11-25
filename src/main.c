#include "image.h"
#include "gaussian.h"

#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, processes;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (argc != 4) {

        if (rank == 0) {
            printf("Usage: ./blur <mode> <input.ppm> <output.ppm>\n");
            printf("Modes: seq, omp, mpi, hybrid\n");
        }

        MPI_Finalize();
        return 1;
    }

    const char *mode = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];

    int valid_mode =
        strcmp(mode, "seq") == 0 ||
        strcmp(mode, "omp") == 0 ||
        strcmp(mode, "mpi") == 0 ||
        strcmp(mode, "hybrid") == 0;

    if (!valid_mode) {
        if (rank == 0)
            fprintf(stderr, "Invalid execution mode.\n");

        MPI_Finalize();
        return 1;
    }

    if ((strcmp(mode, "seq") == 0 ||
         strcmp(mode, "omp") == 0) &&
         processes != 1) {

        if (rank == 0)
            fprintf(stderr, "Sequential and OpenMP require -np 1.\n");

        MPI_Finalize();
        return 1;
    }

    Image input = {0, 0, 3, NULL};
    Image output = {0, 0, 3, NULL};

    if (rank == 0) {

        input = load_image(input_file);

        if (!input.data) {
            fprintf(stderr, "Image loading failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Input image: %d x %d\n",
               input.width, input.height);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    if (strcmp(mode, "seq") == 0) {

        if (rank == 0)
            gaussian_sequential(&input, &output);

    }
    else if (strcmp(mode, "omp") == 0) {

        if (rank == 0)
            gaussian_openmp(&input, &output);

    }
    else if (strcmp(mode, "mpi") == 0) {

        gaussian_distributed(&input, &output, 0);

    }
    else if (strcmp(mode, "hybrid") == 0) {

        gaussian_distributed(&input, &output, 1);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    double local_time = end - start;
    double execution_time = 0.0;

    MPI_Reduce(
        &local_time,
        &execution_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        if (!output.data) {
            fprintf(stderr, "Image processing failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        save_image(output_file, &output);

        printf("\n========== GAUSSIAN BLUR ==========\n");

        printf("Mode: %s\n", mode);
        printf("MPI Processes: %d\n", processes);
        printf("OpenMP Threads: %d\n",
               strcmp(mode, "omp") == 0 ||
               strcmp(mode, "hybrid") == 0
                   ? omp_get_max_threads() : 1);

        printf("Execution Time: %.6f seconds\n",
               execution_time);

        printf("Output: %s\n", output_file);

        printf("==================================\n");
    }

    free_image(&input);
    free_image(&output);

    MPI_Finalize();

    return 0;
}
