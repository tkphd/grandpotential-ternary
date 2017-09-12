CXX = g++
MPICXX = mpicxx
CFLAGS = -O3 -Wall -pedantic -I$(MMSP_PATH)/include -std=c++11
LINKS = -fopenmp -lm -lz

run: PFGP
	./PFGP --example 2 data/test.dat && ./PFGP data/test.dat 100000 1000
.PHONY: run

PFGP: phaseFieldGP.cpp phaseFieldGP.hpp
	$(CXX) $(CFLAGS) $< -o $@ $(LINKS)

parallel: phaseFieldGP.cpp phaseFieldGP.hpp
	$(MPICXX) $(CFLAGS) -include mpi.h $< -o $@ $(LINKS)

.PHONY: clean
clean: cleanobjects

.PHONY: cleanall
cleanall: cleanobjects cleanoutputs

.PHONY: cleanobjects
cleanobjects:
	rm -f parallel PFGP

.PHONY: cleanoutputs
cleanoutputs:
	rm -f data/*.dat data/*.pvd data/*.vti energy.csv energy.png velocity.png
