
CC = mpicc

CFLAGS = -O3 -Wall -Wextra -fopenmp -std=c11

LDFLAGS = -fopenmp -lm

SRC = src/main.c \
      src/image.c \
      src/gaussian.c \
      src/distributed.c

TARGET = blur

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
