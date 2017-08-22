CXX = g++
CFLAGS = -O3 -I$(MMSP_PATH)/include -std=c++11
LINKS = -fopenmp -lm -lz

run: PFGP
	./PFGP --example 2 data/test.0000.dat && ./PFGP data/test.0000.dat 1000 100

PFGP: phaseFieldGP.cpp
	$(CXX) $(CFLAGS) $< -o $@ $(LINKS)

clean:
	rm PFGP

cleanall:
	$(MAKE) clean; rm data/*.png data/*.dat data/*.csv
