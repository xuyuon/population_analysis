import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt

#filename = 'output.h5'
filename = 'n_100_output.h5'

reader = emcee.backends.HDFBackend(filename)


range=[(-1, 4), (0, 11),(40, 150)]

figure = corner.corner(reader.get_chain(flat=True), labels=['alpha', 'beta', 'm_min', 'm_max'], range = range)
plt.show()