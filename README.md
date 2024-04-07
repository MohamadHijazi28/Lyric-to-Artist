# Lyric-to-Artist

design, construct, and train an RNN model that classifies input lyric (a sequence of words) to the artist that composed it.

Methodology:
1. Use LSTM to process the song as a sequence of words. The class torch.nn.LSTM incorporates all the options learned in class. Use the cell type that performs best for this task.

3. First, tokenize the input sequence using pytorch tokenizer. To do this, use torch.nn.Embedding(𝑁,𝐸), where 𝑁 is the number of tokens in the entire vocabulary, and 𝐸 is the dimension to which you embed the tokens to. This creates a trainable layer of embedding.
The input of the embedding class, once constructed, is a Tensor of 𝑛 integer indices.

For example:
Text = “this sentence is long. This is a long sentence”
Tensor of indices = (1, 2, 3, 4, 1, 3, 5, 4, 2)
And the tensor can be given as input to the embedding layer.
You can download and use a pretrained embedding layer, instead
of training your own as a trainable layer.

Data:
CSV files songdata_train.csv, songdata_test.csv are available in the
assignment box; load them into your Notebook.
• The python file LyricsDataset.py contains a class you can use as datasets for loading the files into the workspace.
• You can use torch class DataLoader() to construct a data loader object for easily iterating through the LyricsDataset dataset
