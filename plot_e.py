import pickle
import matplotlib.pyplot as plt

relu_filename = 'train_validation_result_ReLU_lr5e_05_epoch600_esepoch3.pkl'
sigmoid_filename = 'train_validation_result_sigmoid_lrp0001_epoch600_esepoch3.pkl'

result_sigmoid = pickle.load(open(sigmoid_filename, 'rb'))
result_relu = pickle.load(open(relu_filename, 'rb'))
savefig = True

# plot loss for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_sigmoid['epoch'], result_sigmoid['train_err'], color='blue', label='Train loss')
plt.plot(result_sigmoid['epoch'], result_sigmoid['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('sig_loss.pdf', dpi=300)
plt.show()

# plot loss for relu
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_relu['epoch'], result_relu['train_err'], color='blue', label='Train loss')
plt.plot(result_relu['epoch'], result_relu['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('relu_loss.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_sigmoid['epoch'], result_sigmoid['train_acc'], color='blue', label='Train loss')
plt.plot(result_sigmoid['epoch'], result_sigmoid['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('sig_acc.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_relu['epoch'], result_relu['train_acc'], color='blue', label='Train loss')
plt.plot(result_relu['epoch'], result_relu['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('relu_acc.pdf', dpi=300)
plt.show()

