NVCC ?= nvcc
TARGET := part_a_wave
SRC := part_a_wave.cu
NVCCFLAGS := -O3 -lineinfo

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
