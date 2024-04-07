import os
import string

import torch
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from nltk import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext
from LyricsDataset import LyricsDataset
from nltk.corpus import stopwords
from torch.optim.lr_scheduler import ReduceLROnPlateau


def artist_number(artist_name):
    artist_name_number = {"ABBA": 0, "Bee Gees": 1, "Bob Dylan": 2, }
    return artist_name_number.get(artist_name, -1)  # Return -1 if artist name not found


stop_words = stopwords.words('english')


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = ''.join(char for char in text if char not in string.punctuation)
    text = text.replace('\n', ' ')
    text = text.replace('   ', ' ')
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Function to collate data samples into batches
def batch_data(batch):
    _, lyrics, _, artists = zip(*batch)
    artists_tensor = torch.tensor([artist_number(artist) for artist in artists], dtype=torch.long)
    lyrics = [preprocess_text(lyric) for lyric in lyrics]
    lyrics_tensor = [torch.tensor([lyrics_counter[token] for token in lyric], dtype=torch.long) for lyric in lyrics]
    lyrics_tensor = pad_sequence(lyrics_tensor, padding_value=0, batch_first=True)
    return lyrics_tensor, artists_tensor


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.ModuleList([
            nn.LSTM(embedding_dim, hidden_dim, batch_first=True),
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        for ls_tm in self.lstm:
            embedded, _ = ls_tm(embedded)
        output = self.fc(embedded[:, -1, :])
        return output


def train_model(model, train_loader, train_criterion, train_optimizer, num_epochs=10):
    train_loss_values = []
    train_accuracy_values = []
    test_loss_values = []
    test_accuracy_values = []

    best_acc = 0.0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        for lyrics, artists in train_loader:
            train_optimizer.zero_grad()
            output = model(lyrics)
            loss = train_criterion(output, artists)
            loss.backward()
            train_optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_samples += artists.size(0)
            correct += (predicted == artists).sum().item()
            train_loss_values.append(loss.item())
            train_accuracy_values.append(correct / total_samples)

            train_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {train_loss:.4f}, Accuracy: {(correct / total_samples) * 100:.2f}%')

            test_loss, test_accuracy = test_model(model, test_loader)
            test_loss_values.append(test_loss)
            test_accuracy_values.append(test_accuracy)

        test_loss, test_accuracy = test_model(model, test_loader)
        test_loss_values.append(test_loss)
        test_accuracy_values.append(test_accuracy)

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model = model.state_dict()

    plt.figure(figsize=(10, 5))

    # Plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values, label='Training Loss')
    plt.plot(test_loss_values, label='Test Loss')
    plt.xlabel('Global Batch Steps')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig('Train_test_loss.png')
    plt.show()

    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_values, label='Training Accuracy')
    plt.plot(test_accuracy_values, label='Test Accuracy')
    plt.xlabel('Global Batch Steps')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.savefig('train_test_accuracy.png')
    plt.show()

    #plt.tight_layout()
    plt.close()

    # Save the best model
    torch.save(best_model, 'best_model.pth')


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for lyrics, artists in test_loader:
            output = model(lyrics)
            loss = criterion(output, artists)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += artists.size(0)
            correct += (predicted == artists).sum().item()

    accuracy = correct / total
    test_loss = total_loss / len(test_loader)
    print(f"Test Accuracy: {accuracy}")

    return test_loss, accuracy


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('device: ', device)

# Load the datasets
train_dataset = LyricsDataset("songdata_train.csv")
test_dataset = LyricsDataset("songdata_test.csv")

# Tokenization and vocabulary creation
tokenizer = get_tokenizer("basic_english")

lyrics_counter = Counter()
for lyrics in train_dataset.get_vocab():
    lyrics_counter.update(tokenizer(lyrics))


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=batch_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=batch_data)

num_artists = len(set(artist_number(artist) for _, _, _, artist in train_dataset))
model = LSTMModel(len(lyrics_counter), embedding_dim=256, hidden_dim=128, output_dim=num_artists)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train_model(model, train_loader, criterion, optimizer, num_epochs=100)
# test_model(model, test_loader)


# Load the saved model
saved_model_path = 'best_model.pth'
model.load_state_dict(torch.load(saved_model_path))

# Evaluate the loaded model
test_model(model, test_loader)

print(f'parameters number : {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
