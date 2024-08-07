import torch
from torch.utils.data import Dataset, DataLoader
from fetch_data import fetch_documents
from sklearn.preprocessing import LabelEncoder
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW

class ConversationDataset(Dataset):
    def __init__(self, documents, tokenizer, max_length=128):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self._prepare_data()

    def _prepare_data(self):
        self.texts = []
        self.labels = []

        for doc in self.documents:
            for utterance in doc.get('utterances', []):
                text = utterance.get('text')
                role = utterance.get('role')
                if text and role:
                    self.texts.append(text)
                    self.labels.append(role)
        
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# documents = fetch_documents()
# train = fetch_documents('empathy_train')
# dataset = ConversationDataset(train, tokenizer)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# len(dataloader)

# for input_ids, attention_mask, labels in dataloader:
#     print(input_ids, attention_mask, labels)


# from dataset import ConversationDataset
def train_model(model, train_loader, valid_loader, optimizer, device='cuda', epochs=3):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_train_loss}')

        model.eval()
        total_eval_loss = 0
        for batch in valid_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
        avg_val_loss = total_eval_loss / len(valid_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# documents = fetch_documents()
train = fetch_documents('empathy_train', 40)
valid = fetch_documents('empathy_validation', 40)

# 데이터 확인을 위해 일부 문서 출력
print(f'Number of training documents: {len(train)}')
print(f'Number of validation documents: {len(valid)}')

train_dataset = ConversationDataset(train[:int(len(train)*0.8)], tokenizer)
valid_dataset = ConversationDataset(valid[int(len(valid)*0.8):], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

print(len(train_loader), len(valid_loader))

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_dataset.labels)))
optimizer = AdamW(model.parameters(), lr=5e-5)

train_model(model, train_loader, valid_loader, optimizer)
