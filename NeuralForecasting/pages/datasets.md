# Datasets
***

The motor neural activity forecasting dataset includes recorded neural signals from two monkeys performing reaching activities, Monkey A and Monkey B, using $\mu$ECoG arrays. Recorded neural signals are in the form of multivariate continuous time series, with variables corresponding to recording electrodes in the recording array. 


The dataset includes all 239 electrodes from Monkey A and 87 electrodes specifically from the M1 region of Monkey B. 

------
### Dataset format 

The dataset provided follows the shape: 

**Neural_data:  N * T * C * F ( Sampe_size * Time_steps * Channel * Feature )**

**Sampe_size**: varies depending on the dataset. The exact number is summarized in the next section

**Time_steps**: Each sample will have 20 time steps recorded. The model is expected to take the first 10 steps as input and predict the following 10 steps. 

**Channel**: The number of electrodes, which depends on the Monkey. 239 electrodes from Monkey A and 87 electrodes from Monkey B

**Feature**: There are nine features provided. The first feature ([0]) is the final prediction we want the model to take as input and predict. All the remaining features ([1:]) are the decomposition of the original feature in different frequency bands. 




-------

### Training dataset

We provide 
- 985 training samples for Monkey A (affi) and 
- 700 training samples for Monkey B (beignet)

Additional sample records from different dates were provided. 
- 162 training sample records from Monkey A and
- 82 + 76 training sample records from Monkey B


### Testing data

A hold-out dataset is used to evaluate model performance on Codabench.

- 122 + 162 samples from Monkey A
- 87 + 82 + 76 samples from Monkey B



### Final secret dataset 

Another set of secret datasets will be used to evaluate the final ranking of the competition.

### **How to get Public Data?**

Download the [NSF_HDR_Neuro_dataset](FIXME)

Or use the following command to download using the terminal.

```
wget -O public_data.zip FIXME
```