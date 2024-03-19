import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = 'project_results.txt'

data = pd.read_csv(file_path, sep=';')

x_column = 'Epochs'
y_column = 'Accuracy'
z_column = 'Learning Rate'
color_tab = 'MixUp'
params_col = 'Parameters'
amount = 'Amount'

# Extract relevant columns
epochs = data['Epochs']

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return float(value.replace('[', '').replace(']', '').split(',')[0])  # Assuming the first value is correct

train_losses = data['Train Losses'].apply(lambda x: [convert_to_float(val) for val in x.strip('[]').split(',')])
test_losses = data['Test Losses'].apply(lambda x: [convert_to_float(val) for val in x.strip('[]').split(',')])
accuracies = data['Accuracies'].apply(lambda x: [convert_to_float(val) for val in x.strip('[]').split(',')])

epochs = range(1,epochs[1]+1)

plt.figure(1)  
# plt.plot(epochs, train_losses.iloc[1], label='Train Losses', color='blue', alpha=0.5)
# plt.plot(epochs, test_losses.iloc[1], label='Test Losses', color='red', alpha=0.5)


plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Losses over Epochs')
plt.legend()
plt.grid(True)
# plt.show()
# plt.savefig('images/Project/PreAct_100_DA_losses.png')

plt.figure(2)
# plt.plot(epochs, accuracies.iloc[1], label='Accuracies')

plt.title('Accuracy over Epochs')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.grid(True)
# plt.show()
# plt.savefig('images/Project/PreAct_100_DA_accuracies.png')
