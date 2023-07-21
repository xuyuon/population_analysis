import emcee
import numpy as np
import corner
import matplotlib.pyplot as plt

#filename = 'output.h5'
filename = 'output.h5'

reader = emcee.backends.HDFBackend(filename)

range=[(-1, 4),(-1, 3), (0, 10),(40, 100)]
figure = corner.corner(reader.get_chain(flat=True), labels=['alpha', 'beta', 'm_min', 'm_max'], quantiles=(0.16, 0.84),show_titles=True, title_fmt = '.2f', use_math_text=True, range = range)
#figure = corner.corner(reader.get_chain(flat=True), labels=['alpha', 'beta', 'delta', 'm_min', 'm_max', 'lambda', 'mu', 'sigma'], quantiles=(0.16, 0.84),show_titles=True, title_fmt = '.2f', use_math_text=True)

plt.show()