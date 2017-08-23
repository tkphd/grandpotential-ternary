CXX = g++
CFLAGS = -O3 -Wall -pedantic -I$(MMSP_PATH)/include -std=c++11
LINKS = -fopenmp -lm -lz

all: run

.PHONY: all run clean cleanobjects cleanoutputs cleanall

run: PFGP
	./PFGP --example 2 data/test.dat && ./PFGP data/test.dat 100000 10000

PFGP: phaseFieldGP.cpp
	$(CXX) $(CFLAGS) $< -o $@ $(LINKS)

clean: cleanobjects

cleanall: cleanobjects cleanoutputs

cleanobjects:
	rm PFGP

cleanoutputs:
	rm data/*.pvd data/*.vti energy.csv
