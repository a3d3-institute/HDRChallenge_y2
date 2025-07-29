# Starting kit and example submission
***
## Starting kit 
A Google Colab notebook is provided for the participant to explore and train new ML models. The participant is encouraged to copy and make modifications for their training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](FIXME)

 ### ⬇️ [Example  Submission](https://github.com/a3d3-institute/HDRChallenge_y2/blob/main/NeuralForecasting/example_submission/simple_torch_submission/simple_torch_submission.zip)

Models must be trained on up-to-date versions of TensorFlow/PyTorch/Scikit-learn/etc. An example of the intended format of a submission is:

```
import torch
import os

class Model:
 def __init__(self, monkey_name=""):
 # You could include a constructor to initialize your model here, but all calls will be made to the load method
 self.monkey_name = monkey_name
 if self.monkey_name == 'beignet':
 # Load setting for beignet models 
 self.input_size = 89
 elif self.monkey_name == 'affi':
 # Load setting for affi models 
 self.input_size = 239
 else:
 raise ValueError(f'No such a monkey: {self.monkey_name}')
            
 def predict(self, X):
 # This method should accept an input of any size (of the given input format) and return predictions appropriately with the shape (Sample_size * Time_steps * Channel)

 return # Something

 def load(self):
 # This method should load your pre-trained model from wherever you have it saved
 path = "model.pth"
 if self.monkey_name == 'beignet':
 path = "model_beignet.pth"
 elif self.monkey_name == 'affi':
 path = "model_affi.pth"
 else:
 raise ValueError(f'No such a monkey: {self.monkey_name}')
 self.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), path), weights_only=True))
```

The essential functions are ```predict``` and ```load```, the former should return an array of predicted probabilities in the shape of (Sample_size * 20 * Channel). The latter should load a pre-trained model; any auxiliary files necessary, such as "config.json" and "model.weights.h5", should also be included in the submission. The submission will be a zipped file (or files). The only required file to be included in any submission is one of the above formats, named "model.py". There is no restriction on library usage. Inform the administrators if the library you wish to use is not found by open a issue at [GitHub](https://github.com/a3d3-institute/HDRChallenge_y2/issues).


### Expected model inputs

The model will take an **numpy array** of shape **(Sampe_size * 20 * Channel * Feature)**

Only the first 10 steps have meaningful value. The last 10 steps are masked and repeat the 10th step's values. 

### Expected model outputs & target


The model will be expected to return a **numpy** array of shape **(Sample_size * 20 * Channel)**, which should contain the first 10 steps and the predicted next 10 steps. 

Only the first feature is the target for the prediction. All the remaining features ([1:]) are the decomposition of the original feature into different frequency bands. The participants are free to choose whether to use them or not.  



### Common Error

[!!] Do not zip the whole folder. ONLY select the `model.py` and relevant weight and requirements files to make the tarball.

<img src="https://github.com/user-attachments/assets/10b49a84-d42a-42c2-8855-e4b563b28b15" alt="common_error: no module named model" width="750">

The above error is most likely caused by zipping the whole folder (instead of zipping just the contents) when making the tarball.

[!!] Relative path to weight files or any files would not work on Codabench. Please use the example above in the example submission for the weight files path.