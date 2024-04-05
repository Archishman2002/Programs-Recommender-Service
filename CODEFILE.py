import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

#LOADING JSON DATA
with open('/content/program.json', 'r') as file:
    data = json.load(file)

# Extract features and labels
descriptions = [entry["program_desc"] for entry in data]
titles = [entry["program_title"] for entry in data]
levels = [entry["program_level"] for entry in data]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    descriptions,
    levels,
    test_size=0.2,
    random_state=42  # Add a random seed for reproducibility
)

#TOKENISING THE TEXT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(title, description):
    return tokenizer(title + " " + description, return_tensors="pt", truncation=True, padding=True)

# Tokenize training data
X_train_tokens = [tokenize_data(title, desc) for title, desc in zip(X_train, X_train)]
X_test_tokens = [tokenize_data(title, desc) for title, desc in zip(X_test, X_test)]

#CREATING DATALOADER
from torch.nn.utils.rnn import pad_sequence

# Tokenize training data
X_train_tokens = [tokenize_data(title, desc) for title, desc in zip(X_train, X_train)]

# Pad tokenized sequences to the same length
input_ids_padded = pad_sequence([x['input_ids'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = pad_sequence([x['attention_mask'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = attention_mask_padded.squeeze()

# Convert y_train to numerical format
label_dict = {label: i for i, label in enumerate(set(y_train))}
y_train_numeric = torch.tensor([label_dict[label] for label in y_train])

# Create TensorDataset
train_data = TensorDataset(input_ids_padded, attention_mask_padded, y_train_numeric)

# Create DataLoader
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(y_train)))
optimizer = AdamW(model.parameters(), lr=5e-5)

#MODEL TRAINING
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss}")

#MODEL EVALUATION
from sklearn.preprocessing import LabelEncoder

# Convert string labels to numerical values
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

model.eval()

# Pad tokenized sequences to the same length
input_ids_padded_test = pad_sequence([x['input_ids'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = pad_sequence([x['attention_mask'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = attention_mask_padded_test.squeeze()

# Create TensorDataset
test_data = TensorDataset(input_ids_padded_test, attention_mask_padded_test, torch.tensor(y_test_encoded))  # Use y_test_encoded

# Create DataLoader
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

predictions = []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

# Convert logits to predicted labels
predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]

# Decode numerical labels back to original strings if needed
predicted_labels_str = label_encoder.inverse_transform(predicted_labels)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy: {accuracy}")

#ACCURACY EVALUATION AFTER FIRST FINE TUNING
from transformers import AdamW, get_linear_schedule_with_warmup

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluate accuracy after fine-tuning
from sklearn.preprocessing import LabelEncoder

# Convert string labels to numerical values
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

model.eval()

# Pad tokenized sequences to the same length
input_ids_padded_test = pad_sequence([x['input_ids'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = pad_sequence([x['attention_mask'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = attention_mask_padded_test.squeeze()

# Create TensorDataset
test_data = TensorDataset(input_ids_padded_test, attention_mask_padded_test, torch.tensor(y_test_encoded))  # Use y_test_encoded

# Create DataLoader
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

predictions = []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

# Convert logits to predicted labels
predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]

# Decode numerical labels back to original strings if needed
predicted_labels_str = label_encoder.inverse_transform(predicted_labels)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy: {accuracy}")

#FINE TUNING - 2
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import torch

# Assuming X_train and y_train are your training data and labels

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit and transform labels
y_train_encoded = label_encoder.fit_transform(y_train)

# Load pre-trained BERT model and tokenizer
num_labels = len(label_encoder.classes_)  # Dynamically get the number of unique labels
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and preprocess your labeled dataset
train_inputs = tokenizer(X_train, truncation=True, padding=True, max_length=128, return_tensors="pt")
train_labels = torch.tensor(y_train_encoded)  # Use the encoded labels

# Create DataLoader
train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

# Fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert_model")
pip install optuna

#HYPERPARAMETER TUNING USING OPTUNA
import optuna
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())

    return accuracy_score(y_test, predictions)

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 3, 10)

    # Use the suggested hyperparameters in the fine-tuning code
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

    # Fine-tune the model with the suggested hyperparameters
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            # Extract input_ids, attention_mask, and labels from the batch
            input_ids_batch = batch[0].to(device)  # Assuming input_ids is at the first index
            attention_mask_batch = batch[1].to(device)  # Assuming attention_mask is at the second index
            labels = batch[2].to(device)  # Assuming labels are at the third index

            # Forward pass
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Evaluate and return the metric of interest (e.g., accuracy)
    accuracy = evaluate_model(model, test_dataloader)  # Implement your evaluation function
    return accuracy

# Perform hyperparameter tuning with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Adjust the number of trials as needed
best_params = study.best_params

accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy: {accuracy}")

#DATA PREPROCESSING
from torch.utils.data import WeightedRandomSampler

# Assuming you have imbalanced labels in your dataset
class_weights = [1.0, 2.0, 0.5]  # Adjust according to your dataset

# Map labels to integers
label_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}  # Update with your label mapping
numeric_labels = [label_mapping[label] for label in y_train]

weights = [class_weights[label] for label in numeric_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Create a DataLoader with the weighted sampler
train_dataloader = DataLoader(train_data, batch_size=8, sampler=sampler)

#REGULARISATION
from torch.nn import Dropout

# Add dropout layer to the BERT model
model.bert.encoder.layer[-1].output.dropout = Dropout(p=0.1)

#SCRIPT TO RUN - BERT API ENDPOINTS CREATION
<WRITE THIS WEB SCRAPPING CODE YOURSELF>

#FURTHER OPTIMISATION AND FINE TUNING WITH REDUCED LEARNING RATE
pip install numpy

# Loading necessary libraries again and setting the device
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import optuna
import numpy as np

# Loading and preprocessing the data
with open('/content/program.json', 'r') as file:
    data = json.load(file)

descriptions = [entry["program_desc"] for entry in data]
titles = [entry["program_title"] for entry in data]
levels = [entry["program_level"] for entry in data]

X_train, X_test, y_train, y_test = train_test_split(
    descriptions,
    levels,
    test_size=0.2,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(title, description):
    return tokenizer(title + " " + description, return_tensors="pt", truncation=True, padding=True)

X_train_tokens = [tokenize_data(title, desc) for title, desc in zip(X_train, X_train)]
X_test_tokens = [tokenize_data(title, desc) for title, desc in zip(X_test, X_test)]

input_ids_padded = pad_sequence([x['input_ids'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = pad_sequence([x['attention_mask'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = attention_mask_padded.squeeze()

label_dict = {label: i for i, label in enumerate(set(y_train))}
y_train_numeric = torch.tensor([label_dict[label] for label in y_train])

train_data = TensorDataset(input_ids_padded, attention_mask_padded, y_train_numeric)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

# Defining the model again
class CustomBERTModel(BertForSequenceClassification):
    def __init__(self, config):
        super(CustomBERTModel, self).__init__(config)
        # Add your custom modifications or layers if needed

model = CustomBERTModel.from_pretrained("bert-base-uncased", num_labels=len(set(y_train)))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop for optimisation
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss}")

# Evaluating the accuracy after optimisation
model.eval()
input_ids_padded_test = pad_sequence([x['input_ids'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = pad_sequence([x['attention_mask'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = attention_mask_padded_test.squeeze()

test_data = TensorDataset(input_ids_padded_test, attention_mask_padded_test, torch.tensor(y_test_encoded))
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

predictions = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]
predicted_labels_str = label_encoder.inverse_transform(predicted_labels)
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy after optimization: {accuracy}")

# Fine-tuning with reduced learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)  # Adjust the learning rate as needed
num_epochs_fine_tune = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs_fine_tune)

for epoch in range(num_epochs_fine_tune):
    model.train()
    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluating the accuracy after fine-tuning with reduced learning rate
model.eval()
predictions = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]
predicted_labels_str = label_encoder.inverse_transform(np.array(predicted_labels).ravel())
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy after fine-tuning with reduced learning rate: {accuracy}")

#Implementing a Learning Rate Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Replace the existing scheduler with ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

#Implementing Early Stopping
from sklearn.model_selection import StratifiedKFold

# Integrate early stopping into your training loop
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss}")

    # Evaluate on the validation set
    model.eval()
    val_loss = 0

    # Assuming you have a separate validation set (X_val, y_val)
    X_val_tokens = [tokenize_data(title, desc) for title, desc in zip(X_val, X_val)]

    # Pad tokenized sequences to the same length for validation set
    input_ids_padded_val = pad_sequence([x['input_ids'].squeeze() for x in X_val_tokens], batch_first=True)
    attention_mask_padded_val = pad_sequence([x['attention_mask'].squeeze() for x in X_val_tokens], batch_first=True)
    attention_mask_padded_val = attention_mask_padded_val.squeeze()

    # Convert y_val to numerical format
    y_val_numeric = torch.tensor([label_dict[label] for label in y_val])

    # Create TensorDataset for validation set
    val_data = TensorDataset(input_ids_padded_val, attention_mask_padded_val, y_val_numeric)

    # Create DataLoader for validation set
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)

    for input_ids, attention_mask, labels in val_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Validation Loss: {avg_val_loss}")

    # Implement early stopping
    scheduler.step(avg_val_loss)
    if scheduler.state_dict['_last_lr'][0] < 1e-8:
        print("Early stopping as learning rate became too small.")
        break

#IMPLEMENTING ENSEMBLE LEARNING
# Train multiple models and average their predictions
num_models = 5
ensemble_predictions = []

for i in range(num_models):
    # ... (training loop for each model)

    # Use the model to predict on the test set
    model.eval()
    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions.extend(logits.cpu().numpy())

    ensemble_predictions.append(predictions)

# Average predictions from all models
final_predictions = np.mean(ensemble_predictions, axis=0)
final_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in final_predictions]
final_labels_str = label_encoder.inverse_transform(np.array(final_labels).ravel())
accuracy = accuracy_score(y_test, final_labels_str)
print(f"Ensemble Accuracy: {accuracy}")

#Implementing Gradient Accumulation
# Adjust your training loop for gradient accumulation
accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

for epoch in range(epochs):
    model.train()
    total_loss = 0
    accumulation_step_count = 0

    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        accumulation_step_count += 1

        if accumulation_step_count == accumulation_steps:
            # Update weights and reset accumulation
            optimizer.step()
            accumulation_step_count = 0

    # Update weights for the remaining accumulated gradients
    if accumulation_step_count > 0:
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss}")

#RECALCULATING ACCURACY AFTER FINE TUNING AGAIN
# Loading necessary libraries again and setting the device
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import optuna
import numpy as np

# Loading and preprocessing the data
with open('/content/program.json', 'r') as file:
    data = json.load(file)

descriptions = [entry["program_desc"] for entry in data]
titles = [entry["program_title"] for entry in data]
levels = [entry["program_level"] for entry in data]

X_train, X_test, y_train, y_test = train_test_split(
    descriptions,
    levels,
    test_size=0.2,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(title, description):
    return tokenizer(title + " " + description, return_tensors="pt", truncation=True, padding=True)

X_train_tokens = [tokenize_data(title, desc) for title, desc in zip(X_train, X_train)]
X_test_tokens = [tokenize_data(title, desc) for title, desc in zip(X_test, X_test)]

input_ids_padded = pad_sequence([x['input_ids'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = pad_sequence([x['attention_mask'].squeeze() for x in X_train_tokens], batch_first=True)
attention_mask_padded = attention_mask_padded.squeeze()

label_dict = {label: i for i, label in enumerate(set(y_train))}
y_train_numeric = torch.tensor([label_dict[label] for label in y_train])

train_data = TensorDataset(input_ids_padded, attention_mask_padded, y_train_numeric)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

# Defining the model again
class CustomBERTModel(BertForSequenceClassification):
    def __init__(self, config):
        super(CustomBERTModel, self).__init__(config)
        # Add your custom modifications or layers if needed

model = CustomBERTModel.from_pretrained("bert-base-uncased", num_labels=len(set(y_train)))
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop for optimisation
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{epochs} - Average Training Loss: {avg_loss}")

# Evaluating the accuracy after optimisation
model.eval()
input_ids_padded_test = pad_sequence([x['input_ids'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = pad_sequence([x['attention_mask'].squeeze() for x in X_test_tokens], batch_first=True, padding_value=0)
attention_mask_padded_test = attention_mask_padded_test.squeeze()

test_data = TensorDataset(input_ids_padded_test, attention_mask_padded_test, torch.tensor(y_test_encoded))
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)

predictions = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]
predicted_labels_str = label_encoder.inverse_transform(predicted_labels)
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy after optimization: {accuracy}")

# Fine-tuning with reduced learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)  # Adjust the learning rate as needed
num_epochs_fine_tune = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs_fine_tune)

for epoch in range(num_epochs_fine_tune):
    model.train()
    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluating the accuracy after fine-tuning with reduced learning rate
model.eval()
predictions = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions.extend(logits.cpu().numpy())

predicted_labels = [torch.argmax(torch.from_numpy(pred.reshape(1, -1)), dim=1).item() for pred in predictions]
predicted_labels_str = label_encoder.inverse_transform(np.array(predicted_labels).ravel())
accuracy = accuracy_score(y_test, predicted_labels_str)
print(f"Accuracy after fine-tuning with reduced learning rate: {accuracy}")

#END OF THE CODE
