import pickle
import matplotlib.pyplot as plt

half_filename = 'train_validation_result_tanh_lr0.0001_epoch300_esepoch3_hl25.pkl'
double_filename = 'train_validation_result_tanh_lr0.0001_epoch300_esepoch3_hl100.pkl'

result_half = pickle.load(open(half_filename, 'rb'))
result_double = pickle.load(open(double_filename, 'rb'))
savefig = True

# plot loss for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_half['epoch'], result_half['train_err'], color='blue', label='Train loss')
plt.plot(result_half['epoch'], result_half['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('half_loss.pdf', dpi=300)
plt.show()

# plot loss for relu
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_double['epoch'], result_double['train_err'], color='blue', label='Train loss')
plt.plot(result_double['epoch'], result_double['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('double_loss.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_half['epoch'], result_half['train_acc'], color='blue', label='Train loss')
plt.plot(result_half['epoch'], result_half['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('half_acc.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_double['epoch'], result_double['train_acc'], color='blue', label='Train loss')
plt.plot(result_double['epoch'], result_double['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('double_acc.pdf', dpi=300)
plt.show()

