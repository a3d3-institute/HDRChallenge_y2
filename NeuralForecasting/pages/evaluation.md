## Evaluation


Models will be evaluated on a combined metric of $R^2$ and Mean squared error (MSE).  $R^2$ evaluates how predicted future signals can explain the variance in recorded future signals.
MSE measures absolute discrepancy between predicted neural signals and the recorded neural signals.


The provided model is expected to provide the continuous score $p$ of a given event that predicts how likely it's an anomaly. 
The script on Codabench will determine a selection cut, $\hat{p}$,  to avoid participants fine-tuning the selection cut. 

The selection cut is defined at **True Positive Rate (TPR)** of 90% using the evaluation dataset on Codabech. 

For a given event, 
The event is predicted as an anomaly if the $p > \hat{p} $.
The event is predicted as a background if the $p < \hat{p} $.

The final ranking for the leaderboard is determined by the corresponding **True Negative Rate (TNR)**. The higher the score, the better.