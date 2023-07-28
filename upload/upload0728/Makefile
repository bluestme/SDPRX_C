CC := g++
CFLAGS := -Igsl/include -std=c++14
LDFLAGS := -Lgsl/lib
LIBS := -lgsl -lgslcblas -lm

# Target binary name
TARGET := SDPRX

# Source files
SRC := main.cpp mcmc.cpp parse_gen.cpp function_pool.cpp SDPRX_io.cpp LD.cpp

# Object files
OBJ := $(SRC:.cpp=.o)

# Rule to make all
all: $(TARGET)

# Rule to link object files and create the binary
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

# Rule to compile .cpp files into object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean the build files
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean