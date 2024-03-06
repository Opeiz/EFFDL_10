import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'results_presentation.txt'

data = pd.read_csv(file_path, sep=';')

subset = data[data['Model'] == 'PreActResNet18_N']
print(subset)

x_column = 'Epochs'
y_column = 'Accuracy'
z_column = 'Learning Rate'
color_tab = 'MixUp'
params_col = 'Parameters'
train = 'Train Losses'
test = 'Test Losses'
amount = 'Amount'

# train_losses = [eval(losses) for losses in subset[train]]
# test_losses = [eval(losses) for losses in subset[test]]

# epochs = range(1, len(train_losses) + 1)

# plt.plot(epochs, train_losses, label='Train Losses')
plt.scatter(subset['Amount'], subset[y_column], c= subset[amount], cmap='viridis')
plt.xlabel('Epochs') 
plt.ylabel('Losses')
plt.savefig('images/Accuracy vs Amount.png')
plt.legend()