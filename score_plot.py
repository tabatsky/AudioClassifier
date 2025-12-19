import numpy as np
import numpy.core.defchararray as np_f
import csv

import matplotlib.pyplot as plt

from artist_net import version_name

with open(f'{version_name}_3_300_24_accuracy.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    headers = next(reader)
    data = np.array(list(reader))
    data = np_f.replace(data, '\'', '')
    data = data.astype(float)

indices = np.where(data[:, 0] == 0)[0]
last_index = indices[-1]
data = data[last_index:, :]


accum1 = 0.0
accum2 = 0.0
accum_coeff = 0.9
(H, W) = data.shape
for i in range(H):
    accum1 = accum1 * accum_coeff + data[i, 1] * (1 - accum_coeff)
    data[i, 1] = accum1
    accum2 = accum2 * accum_coeff + data[i, 2] * (1 - accum_coeff)
    data[i, 2] = accum2

plt.plot(data[:, 0], data[:, 1], color='blue')
plt.plot(data[:, 0], data[:, 2], color='red')
plt.show()
