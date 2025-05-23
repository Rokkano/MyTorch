CC=gcc
CXX=g++
RM=rm -f
CXXFLAGS=-fsanitize=address -ggdb3 -std=c++20 
LDLIBS=-fsanitize=address

SRCS=
TST_SRCS=tests/tensor/tensor.cc
OBJS=$(subst .cc,.o,$(SRCS))
TST_OBJS=$(subst .cc,.o,$(TST_SRCS))

main: $(OBJS) src/main.cpp
	$(CXX) -o main $(OBJS) $(LDLIBS)

test: $(TST_OBJS)
	$(CXX) -lcriterion -o test $(OBJS) $(TST_OBJS) $(LDLIBS)
	./test

clean:
	rm src/**/*.o main

.PHONY: clean test main