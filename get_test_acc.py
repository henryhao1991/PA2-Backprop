from neuralnet import Neuralnetwork, Layer, load_data, get_accuracy
import pickle

# load data
filename = 'train_validation_result_tanh_lr0.0001_epoch200_esepoch3_dobule_hl.pkl'
result= pickle.load(open(filename, 'rb'))

# reconstruct the model
model = Neuralnetwork(result['config'])
test_data_fname = 'MNIST_test.pkl'
X_test, y_test = load_data(test_data_fname)

layer_no = 0
for layer_idx, layer in enumerate(model.layers):
    if isinstance(layer, Layer):
        layer_no += 1
        layer.w = result['best_weights'][layer_no]
        layer.b = result['best_bias'][layer_no]

# recover the test result
test_logits = model.forward_pass(X_test, y_test)[1]
accuracy = get_accuracy(test_logits, y_test)
print('Test accuracy is %.2f%%' %(accuracy*100))
