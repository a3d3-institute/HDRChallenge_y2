# Overview
***
### Intro 

Understanding the mechanisms of neural activity is important for diagnosing neurological disorders at early stages, devising effective treatment plans, and helping patients regain movement ability \cite{ajiboye2017restoration, vidal1973toward, wolpaw2002brain}. Among the various ways to study these mechanisms, analyzing the dynamics of neural activity offers a unique perspective on how neurons interact to perform specific functions. Such dynamic properties also pave the way for decoding signals into observable behaviors.

However, most existing approaches to learning neural dynamics focus on modeling concurrent (i.e., immediate) neural activity \cite{le2022stndt,pandarinath2018inferring}%\textcolor{blue}{[citation]}, with comparatively little attention paid to predicting future neural dynamics. Predicting the future neural dynamics is challenging, particularly when the observed activity is incomplete, and additional day-to-day or hour-to-hour drifts in the recording array add further variability.

Prior work \cite{li2023amag} %\textcolor{blue}{[citation]} addressed a simplified scenario, estimating future neural activity using training and testing data collected on the same day to avoid the complexities of day-to-day drifts. In this challenge, we extend that dataset to explore the more difficult task of predicting future neural activity across multiple days, capturing the additional variability introduced by these drifts.


### Problem setting: Neural Forecasting
***
We forecast the activations of a cluster of neurons given previous signals from the same cluster. This targets the critical problem of brain-artificial neuron interfaces, and these models can be used in brain-chip interfaces for artificial limb control, amongst many others.

### Challenge target: 
***
#### Learning the Neural Dynamics through Prediction:
Neural activities are recorded in the form of multivariate time series. 
Previous studies investigated neural dynamics using neural activities in a fixed time window \cite{le2022stndt, ye2021representation}. We challenge the participants to propose methods to measure the changes in neural dynamics from recorded neural activity, such that the trained model can predict future given the past neural activities. 

#### Generalization of Predicting Neural Activity in Unseen Sessions:
Validating the trained model on a new recording session poses an additional challenge due to the change of the recorded neuron sets and the change of status of the recording technique \cite{barrese2013failure}. Here we encourage participants to propose methods that have good generalization ability to a new session. The ability to predict neural activities in a new session has great potential to build future low-latency daily use BCIs.


### How to join this challenge?
***
* Go to the "Starting Kit" tab
* Download the "Dummy sample submission" or "sample submission"
* Go to the "My Submissions" tab
* Submit the downloaded file



### How to participate
***
You should submit a zip file `submission.zip` containing a model in a python file `model.py` according to the specification in the "Model Example" page.

### Question about the challenge?
***
Please open an issue in the GitHub page of this challenge. *[Link](https://github.com/a3d3-institute/HDRChallenge_y2/issues)*



### Reference
***

