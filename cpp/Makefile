all: poisson

CXX = g++
CXXFLAGS = -g -Wall -fopenmp -std=c++11 -lstdc++ -I. -march=native -flto -O3 $(OPT_FLAG)
DEPS = poisson_solver.hpp util.hpp
OBJ = poisson_solver.o main.o 

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $(CXXFLAGS) $< 

poisson: $(OBJ)
	$(CXX) -o $@ $(CXXFLAGS) $^ 

.PHONY: disassembly
disassembly: poisson.s

poisson.s: poisson
	objdump -S --disassemble $< > $@

.PHONY: test
test: poisson
	./test.sh

.PHONY: clean
clean:
	rm -f poisson *.o *.s
