# Define the compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -std=c++11 -Iincludes
CXXFLAGS += -I/opt/homebrew/Cellar/nlohmann-json/3.11.3/include

# Define the source files
SRC = src/driver.cc src/nn.cc src/data_aggregation.cc

# Define the object files (same names as source files but with .o extension)
OBJ = $(SRC:.cc=.o)

# Define the executable name
EXEC = nn

# Default target
all: $(EXEC)

# Link the object files to create the executable
$(EXEC): $(OBJ)
	$(CXX) -o $@ $^

# Compile the .cc files to .o files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJ) $(EXEC)
