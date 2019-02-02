import pickle
import matplotlib.pyplot as plt

result = pickle.load(open('train_validation_result_tanh_lr0.0001_epoch300_esepoch3_hl50.pkl', 'rb'))
savefig = True

fig = plt.figure(figsize=(8, 6))
fig.set_facecolor('white')
ax = fig.add_subplot(111)
ln1 = plt.plot(result['epoch'], result['train_err'], color='blue', label='Train loss')
ln2 = plt.plot(result['epoch'], result['valid_err'], color='red', label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)

ax2 = ax.twinx()
ln3 = plt.plot(result['epoch'], result['train_acc'], color='black', label='Train accuracy')
ln4 = plt.plot(result['epoch'], result['valid_acc'], color='green', label='Validation accuracy')
plt.ylabel('Accuracy', fontsize=15)
plt.xlim(0,)

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, bbox_to_anchor=(1, 0.3), loc='lower right', fontsize=15)
if savefig:
    plt.savefig('train_validation_result.pdf', dpi=300)
plt.show()
