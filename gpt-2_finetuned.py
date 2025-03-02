
from transformers import GPT2Tokenizer
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDB dataset (3000 train, 1000 test)
dataset = load_dataset('csv', data_files='IMDB Dataset.csv')
train_subset = dataset["train"].shuffle(seed=42).select(range(7000))
test_subset = dataset["train"].shuffle(seed=42).select(range(2000))
dataset = {"train": train_subset, "test": test_subset}

# Load GPT-2 tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

# Load HookedTransformer GPT-2 model
base_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
# Freeze GPT-2 layers (optional, speeds up training)
for param in base_model.parameters():
    param.requires_grad = False

# Save initial model state
torch.save(model.state_dict(), "gpt2-small-imdb-sentiment.pt")

# Define label mapping
label_map = {"negative": 0, "positive": 1}



class SentimentClassifier(nn.Module):
    def __init__(self, transformer, hidden_dim=768, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(50257, num_classes)  # Maps hidden state → sentiment classes

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids)  # (batch, seq_len, hidden_dim)
        pooled_output = outputs.mean(dim=1)  # Mean pool across sequence length
        logits = self.classifier(pooled_output)  # Shape: (batch_size, 2)
        return logits
    
model = SentimentClassifier(base_model).to(device)

# Save initial model state
torch.save(model.state_dict(), "gpt2-small-imdb-sentiment.pt")


from torch.utils.data import DataLoader, Dataset

class IMDBDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = dataset["review"]
        self.labels = [label_map[label] for label in dataset["sentiment"]]
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label
    

train_dataset = IMDBDataset(dataset["train"], tokenizer)
test_dataset = IMDBDataset(dataset["test"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define optimizer & loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

epochs = 3
model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for input_ids, attention_mask, labels in loop:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        # Forward pass
        logits = model(input_ids,attention_mask)
        loss = criterion(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# Save fine-tuned model
torch.save(model.state_dict(), "gpt2-small-imdb-finetuned.pt")


# Evaluation of the Finetuned model 
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids,attention_mask)
        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


base_model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model = SentimentClassifier(base_model).to(device)

# evaluate the model on the base model only 
model.load_state_dict(torch.load("gpt2-small-imdb-base.pt"))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids,attention_mask)
        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
