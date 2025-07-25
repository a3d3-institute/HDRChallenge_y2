# Starting kit and example submission
***
## Starting kit 
A Google Colab notebook is provided for the participant to explore and train new ML models. The participant is encouraged to copy and make modifications for their own training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hatkYT5Xq6qauDXY6xFrfnGzB66QPsV8?usp=sharing)

 ### ⬇️ [Example  Submission](https://github.com/a3d3-institute/HDRchallenge/blob/main/scripts/example_submissions/trivial_submission/submission.zip)

Models must be trained on up-to-date versions of TensorFlow/PyTorch/Scikitlearn/etc. An example of the intended format of a submission is:

```
import tensorflow as tf
import os

class Model:
    def __init__(self):
        # You could include a constructor to initialize your model here, but all calls will be made to the load method
        self.clf = None 

    def predict(self, X):
        # This method should accept an input of any size (of the given input format) and return predictions appropriately
        b = self.clf.predict(X)

        return [i[0] for i in b]

    def load(self):
        # This method should load your pre-trained model from wherever you have it saved
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as file:
            for line in file:
                self.clf = tf.keras.models.model_from_json(line)
        self.clf.load_weights(os.path.join(os.path.dirname(__file__), 'model.weights.h5'))
```

The essential functions are ```predict``` and ```load```, the former should return an array of predicted probabilities in the range [0, 1]. The latter should load a pre-trained model; any auxiliary files necessary, such as "config.json" and "model.weights.h5", should also be included in the submission. The submission will be a zipped file (or files). The only required file to be included in any submission is one of the above formats, named "model.py". There is no restriction on library usage. Inform the administrators if a library you wish to use is not found


### Common Error

[!!] Do not zip the whole folder. ONLY select the `model.py` and relevant weight and requirements files to make the tarball.

<img src="https://github.com/user-attachments/assets/10b49a84-d42a-42c2-8855-e4b563b28b15" alt="common_error: no module named model" width="750">

The above error is mostly likely caused by zipping the whole folder (instead of zipping just the contents) when making the tarball.

[!!] Relative path to weight files or any files would not work on codabench. Please use the example above in example submission for the weight files path.