CC = g++
CFLAGS = -g -Wall -std=c++14 -Ofast
SRCS = src/shifting.cpp
PROG ?= defaultProg
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -lpthread
INCLUDE = include

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) -I $(INCLUDE)
