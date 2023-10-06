CC := g++
CFLAGS := -Igsl/include -std=c++11 -Wall -Wextra -pedantic -DHAVE_INLINE -march=native
LDFLAGS := -Lgsl/lib/ -Wl,-rpath gsl/lib -LMKL/lib/ -Wl,--no-as-needed,-rpath MKL/lib/
LIBS := -lgsl -lmkl_rt -lm -lpthread -ldl

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
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS) $(LDFLAGS)

# Rule to compile .cpp files into object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean the build files
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
