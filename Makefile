CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g $(shell root-config --cflags)
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS=$(shell root-config --libs)

SRCS=src/main.cpp
OBJS=$(subst .cc,.o,$(SRCS))

main: $(OBJS)
	$(CXX) $(LDFLAGS) -fsanitize=address -ggdb3 -std=c++20 -o main $(OBJS) $(LDLIBS)

clean:
	rm src/**/*.o main

.PHONY: clean