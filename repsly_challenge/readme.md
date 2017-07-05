# Repsly challenge

The goal of this exercise is to predict who will buy the Replsy service, given their trial history.

## Exercise 1

This exercise will check that everything is working properly on you local setup (e.g. Python and Tensorflow version).

The steps are as follows:

1. Unzip the training file `data/trial_users_analysis.txt`.

2. Switch to appropriate python environment. E.g. using Anaconda and environment `tf-1.2` it is as follows:

   ```
   > source activate tf-1.2
   ``` 
   
   assuming the environment name `tf-1.2`. (NB: The code was tested using Python 3.6 and Tensorflow 1.2)
   
3. Change directory and open the Jupyter notebook:

    ```
    > cd deep_learning/repsly_challenge
    > jupyter notebook & 
    ``` 

4. Run the notebook `repsly_challenge.ipynb` from the browser. Depending on the speed of your setup, you should see the following message after
   at most one minute:
   
   ```
    RepslyFC/no_of_layers-6/hidden_size-369/use_batch_norm-True/keep_prob-0.6/input_keep_prob-0.76/batch_norm_decay-0.99/lr-0.001/dr-0.99/ds-20
    RepslyFC/no_of_layers-7/hidden_size-223/use_batch_norm-True/keep_prob-0.67/input_keep_prob-0.66/batch_norm_decay-0.99/lr-0.001/dr-0.99/ds-20
    RepslyFC/no_of_layers-6/hidden_size-258/use_batch_norm-True/keep_prob-0.38/input_keep_prob-0.78/batch_norm_decay-0.99/lr-0.001/dr-0.99/ds-20
    RepslyFC/no_of_layers-5/hidden_size-300/use_batch_norm-True/keep_prob-0.6/input_keep_prob-0.88/batch_norm_decay-0.99/lr-0.001/dr-0.99/ds-20
    RepslyFC/no_of_layers-6/hidden_size-177/use_batch_norm-True/keep_prob-0.42/input_keep_prob-0.81/batch_norm_decay-0.99/lr-0.001/dr-0.99/ds-20
    ################################################################################
    ```

5. Open tensorboard and monitor training using the following command:

   ```
   > tensorboard --logdir graphs/ &
   Starting TensorBoard b'52' at http://localhost:6006
   ```
   Open `http://localhost:6006` in browser and there you can monitor training progress.
   
6. The notebook is setup to cross validate different architectures of the network by varying the number of stacked cells
   and the number of neurons inside individual GRU cells. Unless you are running Tensorflow on GPU, that will take a way
   too long to finish :) The optimal parameters for this dataset are as follows:
   
   ```
    arch = {
            'no_of_layers': 4,
            'hidden_size': 256,
            'use_batch_norm': 'True',
            'keep_prob': 0.68,
            'input_keep_prob': 0.72,
            'batch_norm_decay': 0.99
    }
    learning_dict = {
        'learning_rate': 0.001,
        'decay_steps': 20,
        'decay_rate': 0.99
    }
    train_dict = {
        'batch_size': 512,
        'epochs': 100,
        'skip_steps': 20
    }
   ```
    
7. The model can be used to analyse and predict any kind of sequences. Try feeding it with different data.

## Exercise 2

In this exercise you will learn to build your own network by yourself.

