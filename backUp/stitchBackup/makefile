CC = g++
CFLAGS = -g -Wall -std=c++14 -Ofast
SRCS = src/stitching.cpp src/video_stream.cpp src/utils.cpp
PROG ?= Stitching_nocc
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -lpthread
INCLUDE = include

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) -I $(INCLUDE)
