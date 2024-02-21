import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'results.txt'

data = pd.read_csv(file_path)
print(data)

subset = data[data['Epochs'] == 50].copy()

booleans = subset['MixUp'].unique()

x_column = 'Epochs'
y_column = 'Accuracy'
z_column = 'Learning Rate'
color_tab = 'MixUp'
params_col = 'Parameters'

conditions = [
    subset[color_tab] == booleans[0],
    subset[color_tab] == booleans[1]
]

choices = ['green', 'blue']

colors = np.select(conditions, choices, default='black')
plt.scatter(subset[params_col], subset[y_column], c=colors, alpha=0.5)
plt.xlabel(params_col)
plt.ylabel(y_column)
plt.title(f'{y_column} vs {params_col}')
plt.legend(booleans)
# plt.show()
# plt.savefig('images/Tr