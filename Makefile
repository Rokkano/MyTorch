CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g $(shell root-config --cflags)
LDFLAGS=-g $(shell root-config --ldflags)
LDLIBS=$(shell root-config --libs)

SRCS=src/main.cpp
OBJS=$(subst .cc,.o,$(SRCS))

main: $(OBJS)
	$(CXX) $(LDFLAGS) -o main $(OBJS) $(LDLIBS)