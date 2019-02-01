# CSE253 PA2
1. To check the gradient from numerical and backprop, simply run: ```python check_gradient.py```

2. To train the neural network, simply run: ```python neuralnet.py```

Change `config` accordingly based on different quetions. The test accuracy will be printed, and the train/validation loss/accuracy will be saved in a `.pkl` file.

3. To get the loss/accuracy plot, simply run:
```python plot_<question number>.py```

Each question has one file for plot purpose. Remember to change the filename to generated `.pkl` file.

4. In case the test accuracy is not recorded, we still can get the test accuracy after training. Simply run:
```python get_test_accuracy.py```

Remember to change the filename to the `.pkl` file.
