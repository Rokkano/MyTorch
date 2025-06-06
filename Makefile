CC=gcc
CXX=g++
RM=rm -f
CXXFLAGS=-ggdb3 -std=c++20 
LDLIBS=-fsanitize=address
CRITERIONFLAGS=-fprofile-arcs -ftest-coverage -lcriterion -O0

SRCS=
TST_SRCS=tests/tensor/tensor.cc
OBJS=$(subst .cc,.o,$(SRCS))
TST_OBJS=$(subst .cc,.o,$(TST_SRCS))

main: $(OBJS) src/main.cpp
	$(CXX) $(CXXFLAGS) $(LDLIBS) -o main src/main.cpp $(OBJS)

test: $(TST_OBJS)
	$(CXX) $(CXXFLAGS) $(CRITERIONFLAGS)  $(LDLIBS) -o test $(SRCS) $(TST_SRCS)
	@./test

test-cov: test
	@mkdir -p cov/
	@rm -rf cov/*
	@gcovr --include src/ --json cov/cov.json
	@./scripts/gcov_filelist.sh
	@gcovr --include src/ -a cov/cov_empty.json -a cov/cov.json --html-nested cov/cov.html

benchmark: $(OBJS) scripts/benchmark.cc
	$(CXX) $(CXXFLAGS) -o benchmark scripts/benchmark.cc $(OBJS)
	./benchmark

clean:
	rm -rf $(OBJS) $(TST_OBJS) *.gcda *.gcno main test benchmark cov/

.PHONY: clean test main