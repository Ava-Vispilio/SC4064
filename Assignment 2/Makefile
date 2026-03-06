NVCC ?= nvcc
NVCCFLAGS := -O3 -lineinfo -std=c++17

.PHONY: all clean part_a part_b

all: part_a part_b

part_a:
	$(NVCC) $(NVCCFLAGS) -o part_a_wave part_a_wave.cu

part_b:
	$(NVCC) $(NVCCFLAGS) -o part_b_wave part_b_wave.cu -lcusparse -lcublas

clean:
	rm -f part_a_wave part_b_wave
