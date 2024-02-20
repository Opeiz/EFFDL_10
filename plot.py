import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'results.txt'

data = pd.read_csv(file_path)
print(data)

x_column = 'Epochs'
y_column = 'Accuracy'
z_column = 'Learning Rate'

plt.scatter(data[x_column], data[y_column], cmap='viridis')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.show()