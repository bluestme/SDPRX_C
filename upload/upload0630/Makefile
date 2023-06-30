CC = g++
CFLAGS = -Igsl/include
LDFLAGS = -Lgsl/lib
LIBS = -lgsl -lgslcblas -lm
TARGET = program_name
SRC = test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)