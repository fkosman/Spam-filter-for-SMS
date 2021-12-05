# SMS Message Spam Filter

Before doing anything, you will first need to install NumPy and MatPlot. Enter the following commands into the command line:

```
pip install numpy
pip install matplotlib
```
To train and interact with the models, you will need to enter the "project" directory. Running any of the python programs from outside of "project" will cause issues with saving and loading datasets and other parameters.

```
cd project
```


## Training the model

1. Begin the training process by splitting the dataset into a training and validation set, using the following command:

```
python split_data.py
```


2. Create the input vocabulary by running the following:

```
python vocabulary.py
```
The vocabulary produced will be of size 1,000 by default, based on the most common words encountered in the training dataset, with any words that are in the top 60 frequencies for both spam and on-spam removed. You can edit the size of the vocabulary and of the number of words removed by editing the "vocab_size" and "redundant_vocab_size" constants at the top of vocabulary.py


3. You can now begin training an LSTM model. Enter the following command:

```
python train_model.py
```
You will be prompted to name the model and select its hidden size, as well as the training hyperparameters. Select an appropiate name as it will be used in saving parameters, logging the epoch data, and for evaluation or testing.


## Resuming training

After creating a new model and training it for some number of epochs, you can resume training the same model by simply running  ```python train_model.py``` again and entering the name of the model when prompted. The program will continue training the model where it left off, using the last parameters from the most recent training session, and will continue updating the epoch logs accordingly.


## Testing the model

"test_model.py" allows you to test an existing model with manually entered sample inputs. To use it, run:
```
python test_model.py [MODEL NAME]
```
The name given to the model must be entered in the command line. After running, you can manually enter a string and have the model predict if it is spam or not.


## Plot training data

You can plot the data logged so far for a model's training by running the following command:
```
python plot_epochs.py [MODEL NAME]
```
This will produce two plots, one for the loss and one for the accuracy, with the respect to all epochs the given model has trained over.


## Evaluation against baseline

To compare a model's performance against the validation set along with a baseline, run the following command:
```
python eval_model.py [MODEL NAME]
```
The accuracies, precisions, recalls, and F-scores of the given model will be displayed along with those of the baseline.
The baseline uses a list of keywords to determine if a message is spam. You can choose your own keywords list by opening "eval_model.py" and editing the constructor for the baseline model.
