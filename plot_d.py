import pickle
import matplotlib.pyplot as plt

lambda_p001_filename = 'train_validation_result_tanh_lr0.0001_epoch67_esepoch3_hl50_lambda0.001.pkl'
lambda_p0001_filename = 'train_validation_result_tanh_lr0.0001_epoch360_esepoch3_hl50_lambda0.0001.pkl'

result_lambda_p001 = pickle.load(open(lambda_p001_filename, 'rb'))
result_lambda_p0001 = pickle.load(open(lambda_p0001_filename, 'rb'))
savefig = True

# plot loss for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_lambda_p001['epoch'], result_lambda_p001['train_err'], color='blue', label='Train loss')
plt.plot(result_lambda_p001['epoch'], result_lambda_p001['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('lambda_p001_loss.pdf', dpi=300)
plt.show()

# plot loss for relu
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_lambda_p0001['epoch'], result_lambda_p0001['train_err'], color='blue', label='Train loss')
plt.plot(result_lambda_p0001['epoch'], result_lambda_p0001['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('lambda_p0001_loss.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_lambda_p001['epoch'], result_lambda_p001['train_acc'], color='blue', label='Train loss')
plt.plot(result_lambda_p001['epoch'], result_lambda_p001['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('lambda_p001_acc.pdf', dpi=300)
plt.show()

# plot accuracy for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result_lambda_p0001['epoch'], result_lambda_p0001['train_acc'], color='blue', label='Train loss')
plt.plot(result_lambda_p0001['epoch'], result_lambda_p0001['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('lambda_p0001_acc.pdf', dpi=300)
plt.show()

