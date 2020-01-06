CXXFLAGS=-std=c++17 -Wall -Wextra -pedantic
CXXFLAGS_DEBUG=$(CXXFLAGS) -g3 -O0
CXXFLAGS_SANITIZE=$(CXXFLAGS) -O0 -g3 \
				  -fsanitize=address,undefined -fno-omit-frame-pointer \
				  -fno-optimize-sibling-calls
CXXFLAGS_RELEASE=$(CXXFLAGS) -O3 -march=native -g3
INC=-I span-lite/include -I spqlios/ -I cereal/include
LIB=-L spqlios/build -lspqlios

main: main.cpp aqtfhe.hpp test.inc
	#clang++ $(CXXFLAGS_SANITIZE) -o $@ $< $(INC) $(LIB)
	#clang++ $(CXXFLAGS_DEBUG) -o $@ $< $(INC) $(LIB)
	clang++ $(CXXFLAGS_RELEASE) -o $@ $< $(INC) $(LIB) -lprofiler
