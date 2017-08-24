#!/usr/bin/python

from matplotlib import pylab as plt
import numpy as np

t, f, v = np.loadtxt("energy.csv", skiprows=1, delimiter=',', unpack=True)

plt.figure()
plt.plot(t, f)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.savefig("energy.png", dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t, v)
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.savefig("velocity.png", dpi=400, bbox_inches='tight')
plt.close()
