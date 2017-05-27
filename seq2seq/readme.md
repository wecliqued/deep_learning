# seq2seq model

This model uses stacked GRU cells and it is learning to predict the next character in the sequence.

## Exercise 1

This exercise will check that everything is working properly on you local setup (e.g. Python and Tensorflow version).

The steps are as follows:

1. Download the training file `songs-utf-8.txt` from our Facebook page and place it into the `data/` folder.
   You can later use your own datasets for experiments.

2. Switch to appropriate python environment. E.g. using Anaconda and environment `tf-1.1` it is as follows:

   ```
   > source activate tf-1.1
   ``` 
   
   assuming the environment name `tf-1.1`. (NB: The code was tested using Python 3.6 and Tensorflow 1.1)
   
3. Change directory and open the Jupyter notebook:

    ```
    > cd deep_learning   
    > jupyter notebook & 
    ``` 

4. Run the notebook from the browser. Depending on the speed of your setup, you should see the following message after
   at most one minute:
   
   ```
   Checkpoint directory is: /users/davor/wecliqued/deep_learning/checkpoints/seq2seq-songs-utf-8.txt-stacked_layers3-hidden_size512-window_size128-overlap_size0-lr0.001-dr0.999-ds20
   ################################################################
   [step=0000] loss = 4.7   time = 31.8 sec
   sample: #{-žNK
   zD`=;E|
   +#/&Efx Ić2S$beQ2p;bič eCđLJFQ]4:`_dqAL!P=HI=.Đ
   -]=p!3%kG qqV;KR $u&7j[_yp-cd-;fX(,7{ČIj~%/oO4'@ČlU8!1{Č86h]T3Dmhf7eZj-5#Ž3bO(!Gm@tWtSNč
   ?JIg1}`"vlG	7f&n_č3iO"*=V.9h^RMt.s!=]LćN'E%AČG;a]Gč*Q&Če7đlzl6	@{CWiBžO~r$:`B	<e:GtWž ZW^W`0esnrć&sB0*](0^+PP<n9U:0Mn9{m%K	2-SM
   :jtDS(#tĆ\~ZwYj.;
   ``` 

5. Open tensorboard and monitor training using the following command:

   ```
   > tensorboard --logdir graphs/ &
   Starting TensorBoard b'52' at http://localhost:6006
   ```
   Open `http://localhost:6006` in browser and there you can monitor training progress.
   
6. The notebook is setup to cross validate different architectures of the network by varying the number of stacked cells
   and the number of neurons inside individual GRU cells. Unless you are running Tensorflow on GPU, that will take a way
   too long to finish :) The optimal parameters for this dataset is 3 stacked layers with 512 neurons each, try that if
   you are running on CPU only.
    
7. The model can be used to analyse and predict any kind of sequences. Try feeding it with different data.

## Exercise 2

In this exercise you will learn to build this network by yourself. I highly recommend using unit tests for checking
correctness of the code (you can find them is `test_seq2seq.py`). Probably the simplest way to run them is PyCharm,
but you can also use command line.
 
1. Open `seq2seq.py` in editor and start filling in the missing code blocks surranded by the comment block.

   E.g. the first task is to complete `_create_cell()` function that creates GRU cells:
   
   ```
    def _create_cell(self, seq, no_stacked_cells):
        """
        Creates GRU cell
        :param seq: placeholder of the input batch
        :return: cell and placeholder for its internal state
        """
        batch_size = tf.shape(seq)[0]

        ##########################################################################################################
        #
        # TODO: Create a stacked MultiRNNCell from GRU cells
        #       First, you have to use tf.contrib.rnn.GRUCell() to construct cells
        #       Since around May 2017, there is new way of constructing MultiRNNCell and you need to create
        #       one cell for each layer. Old code snippets that used [cell * no_stacked_cells] that you can
        #       find online might not work with the latest Tensorflow
        #
        #       After construction GRUCell objects, use it to construct tf.contrib.rnn.MultiRNNCell().
        #
        # YOUR CODE BEGIN
        #
        ##########################################################################################################

        cell = None # you

        ##########################################################################################################
        #
        # YOUR CODE END
        #
        ##########################################################################################################

        multi_cell_zero_state = cell.zero_state(batch_size, tf.float32)
        in_state_shape = tuple([None, self.hidden_size] for _ in range(no_stacked_cells))
        in_state = tuple(tf.placeholder_with_default(cell_zero_state, [None, self.hidden_size], name='in_state') for cell_zero_state in multi_cell_zero_state)

        return cell, in_state
   
   ```

    In case you are not sure about how to solve the problem, you can always check the solution in `seq2seq_solution.py`.
    If you have a different solution than the one in the solution file, run unit test to check if it is passing test.
     
  2. After you have completed all functions in step 1., replace the solution  in the notebook file:
     ```
     # Replace model after you finish step 1. in exercise 2.
     # from seq2seq_solution import Seq2Seq
     from seq2seq import Seq2Seq

     ```
 