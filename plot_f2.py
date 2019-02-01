import pickle
import matplotlib.pyplot as plt

filename = 'train_validation_result_tanh_lr0.0001_epoch200_esepoch3_dobule_hl.pkl'

result = pickle.load(open(filename, 'rb'))
savefig = True

# plot loss for sigmoid
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result['epoch'], result['train_err'], color='blue', label='Train loss')
plt.plot(result['epoch'], result['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('twolayers_loss.pdf', dpi=300)
plt.show()

# plot loss for relu
fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
plt.plot(result['epoch'], result['train_acc'], color='blue', label='Train loss')
plt.plot(result['epoch'], result['valid_acc'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)
plt.legend(loc='best')

if savefig:
    plt.savefig('twolayers_acu.pdf', dpi=300)
plt.show()