import numpy as np

"""
Load the checkpoint model that had the best validation loss.
Measure the accuracy of the classification using the test data set. 
"""


def CalculateTestAccuracy(model, test_targets, test_tensors):
    classifications = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # Calculate accuracy
    test_accuracy = 100*np.sum(np.array(classifications) == np.argmax(test_targets, axis=1))/len(classifications)
    print('Test accuracy: %.4f%%' % test_accuracy)
    print(test_accuracy)