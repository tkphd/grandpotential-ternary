#!/usr/bin/python

from matplotlib import pylab as plt
import numpy as np

t, xcr, xnb, f, v = np.loadtxt("energy.csv", skiprows=1, delimiter=',', unpack=True)

plt.figure()
plt.plot(t, f)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.savefig("energy.png", dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t, xcr)
plt.xlabel("Time")
plt.ylabel("Cr")
plt.savefig("mass_Cr.png", dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t, xnb)
plt.xlabel("Time")
plt.ylabel("Nb")
plt.savefig("mass_Nb.png", dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t, v)
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.savefig("velocity.png", dpi=400, bbox_inches='tight')
plt.close()
