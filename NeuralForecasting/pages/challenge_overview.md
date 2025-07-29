# Overview
***
### Intro 

Understanding the mechanisms of neural activity is important for diagnosing neurological disorders at early stages, devising effective treatment plans, and helping patients regain movement abilities [1,2,3]. Among the various ways to study these mechanisms, analyzing the dynamics of neural activity offers a unique perspective on how neurons interact to perform specific functions. Such dynamic properties also pave the way for decoding signals into observable behaviors.

However, most existing approaches to learning neural dynamics focus on modeling concurrent (i.e., immediate) neural activity [4,5], with comparatively little attention paid to predicting future neural dynamics. Predicting the future neural dynamics is challenging, particularly when the observed activity is incomplete, and additional day-to-day or hour-to-hour drifts in the recording array add further variability.

Prior work [6] addressed a simplified scenario by estimating future neural activity using training and testing data collected on the same day to avoid the complexities of day-to-day drifts. In this challenge, we extend that dataset to explore the more difficult task of predicting future neural activity across multiple days, capturing the additional variability introduced by these drifts.


### Problem setting: Neural Forecasting
***
We forecast the activations of a cluster of neurons given previous signals from the same cluster. This targets the critical problem of brain-artificial neuron interfaces, and these models can be used in brain-chip interfaces for artificial limb control, amongst many others.

### Challenge target: 
***
#### Learning the Neural Dynamics through Prediction:
Neural activities are recorded in the form of multivariate time series. 
Previous studies investigated neural dynamics using neural activities in a fixed time window [4,7]. We challenge participants to propose methods to measure the changes in neural dynamics from recorded neural activity, such that the trained model can predict future activities given past neural activities. 

#### Generalization of Predicting Neural Activity in Unseen Sessions:

Validating the trained model on a new recording session poses an additional challenge due to changes in the recorded neuron sets and changes in the status of the recording technique. Here, we encourage participants to propose methods that have good generalization ability to a new session. The ability to predict neural activities in a new session has great potential for building future low-latency daily-use BCIs.


### How to join this challenge?
***
* Go to the "Starting Kit" tab
* Download the "Dummy sample submission" or "sample submission"
* Go to the "My Submissions" tab
* Submit the downloaded file



### How to participate
***
You should submit a zip file `submission.zip` containing a model in a Python file `model.py` according to the specification in the "Model Example" page. The model needs to have an option `monkey_name` to specify which model to train and apply.

### Question about the challenge?
***
Please open an issue on the GitHub page of this challenge. [Link](https://github.com/a3d3-institute/HDRChallenge_y2/issues)



### Reference
***
[1] A Bolu Ajiboye, Francis R Willett, Daniel R Young, William D Memberg, Brian A Murphy, Jonathan P Miller, Benjamin L Walter, Jennifer A Sweet, Harry A Hoyen, Michael W Keith, et al. Restoration of reaching and grasping movements through brain-controlled muscle stimulation in a person with tetraplegia: a proof-of-concept demonstration. The Lancet, 389(10081):1821–1830, 2017.

[2] Jacques J Vidal. Toward direct brain-computer communication. Annual review of Biophysics and Bioengineering, 2(1):157–180, 1973.

[3] Jonathan R Wolpaw, Niels Birbaumer, Dennis J McFarland, Gert Pfurtscheller, and Theresa M Vaughan. Brain–computer interfaces for communication and control. Clinical neurophysiology, 113(6):767–791, 2002.

[4] Trung Le and Eli Shlizerman. Stndt: Modeling neural population activity with spatiotemporal transformers.
Advances in Neural Information Processing Systems, 35:17926–17939, 2022.

[5] Chethan Pandarinath, Daniel J O'Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D Stavisky, Jonathan C
Kao, Eric M Trautmann, Matthew T Kaufman, Stephen I Ryu, Leigh R Hochberg, et al. Inferring single-trial
neural population dynamics using sequential auto-encoders. Nature methods, 15(10):805–815, 2018.

[6] Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, and Eli Shlizerman. Amag:
Additive, multiplicative, and adaptive graph neural network for forecasting neuron activity. Advances in
Neural Information Processing Systems, 36:8988–9014, 2023.

[7] Joel Ye and Chethan Pandarinath. Representation learning for neural population activity with neural data transformers. arXiv preprint arXiv:2108.01210, 2021.
