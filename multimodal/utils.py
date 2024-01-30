
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_scheduler   
import tqdm
from .processor import MultiModalProcessor
import numpy as np
PROCESSOR = MultiModalProcessor()
def collate_fn(batch):
    labels = []
    texts = []
    images = []
    for data in batch:
        labels.append(data['label'])
        texts.append(data['text'])
        images.append(data['image'])
    output = PROCESSOR(texts, images, padding=True, truncation=True, return_tensors="pt")
    input_ids = output['input_ids']
    attention_mask = output['attention_mask']
    token_type_ids = output['token_type_ids']
    images = output['pixel_values']
    
    labels = torch.tensor(labels)
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'images': images, 'labels': labels}
    return batch

# test fine-tuning scripts, unimodal semantic analysis
class MultiModalTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_dataset,
                 eval_dataset = None,
                 test_dataset = None,
                 compute_metrics = None, 
                 batch_size = 1,
                 num_epochs = 3,
                 **kwargs
                 ):
        self.model = model
        
        if eval_dataset is None:
            eval_ratio = kwargs.pop('eval_ratio', 0.1)
            # split train_dataset into train and eval
            eval_size = int(len(train_dataset) * eval_ratio)
            train_size = len(train_dataset) - eval_size
            train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])
        lr = kwargs.pop('lr', 1e-4)  
        self.eval_per_epoch = kwargs.pop('eval_per_epoch', False)
        torch.manual_seed(3307) # 3307 is all you need  
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * len(self.train_loader)
        self.optimizer = AdamW(self.model.parameters(), lr)
        self.lr_scheduler = get_scheduler(
                                name="linear", 
                                optimizer=self.optimizer, 
                                num_warmup_steps=0, 
                                num_training_steps=self.num_training_steps
                            )
        self.compute_metrics = compute_metrics
        pass

    def train(self):
        # init progress bar
        # progress_bar = tqdm(range(self.num_training_steps))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.train()
        self.model.to(device)
        criteration = nn.CrossEntropyLoss()
        
        for epoch in range(self.num_epochs):
            # progress_bar = tqdm(self.train_loader, desc=f"epoch {epoch}/{self.num_epochs}")
            loss_sum = 0
            for _, batch in tqdm.tqdm(enumerate(self.train_loader), desc = f"epoch {epoch}/{self.num_epochs}", total=len(self.train_loader)):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                images = batch['images']
                labels = batch['labels']
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                images = images.to(device)
                labels = labels.to(device)
                # forward
                output = self.model(input_ids, attention_mask, token_type_ids, images)
                loss = criteration(output, labels)
                # backward
                loss.backward()
                loss_sum += loss.item()
                # update
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            print(f"epoch {epoch}/{self.num_epochs}: loss: {loss_sum / len(self.train_loader)}")
            # progress_bar.update(1)
            if self.eval_per_epoch and self.compute_metrics is not None:
                # eval model on eval_dataset
                metrics = self.eval()
                print(f"epoch {epoch}/{self.num_epochs}: {metrics}")
                # reset model to train mode
                self.model.train()
            pass
        if self.compute_metrics is not None:
            metrics = self.eval()
            print(f": {metrics}")
        pass
    def eval(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.eval()
        self.model.to(device)
        progress_bar = tqdm.tqdm(self.eval_loader, leave = False, desc="eval")
        with torch.no_grad():
            labels_true = torch.tensor([]).to(device)
            labels_pred = torch.tensor([]).to(device)
            for batch in self.eval_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                images = batch['images']
                labels = batch['labels']
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                images = images.to(device)
                labels = labels.to(device)
                # forward
                output = self.model(input_ids, attention_mask, token_type_ids, images)
                labels_true = torch.concat((labels_true, labels))
                labels_pred = torch.concat((labels_pred, torch.argmax(output, dim=1)))
                progress_bar.update(1)
            # compute metrics
            # print(labels_true, labels_pred)
            labels_true = labels_true.cpu().numpy()
            labels_pred = labels_pred.cpu().numpy()
            metrics = self.compute_metrics(labels_true, labels_pred)
            return metrics
        pass
    def predict(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.eval()
        self.model.to(device)
        progress_bar = tqdm(self.test_loader)
        with torch.no_grad():
            labels_pred = torch.tensor([]).to(device)
            for batch in self.test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                images = batch['images']
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                images = images.to(device)
                labels = labels.to(device)
                # forward
                output = self.model(input_ids, attention_mask, token_type_ids, images)
                labels_pred = torch.concat((labels_pred, torch.argmax(output, dim=1)))
                progress_bar.update(1)
            # compute metrics
            # print(labels_true, labels_pred)
            labels_pred = labels_pred.cpu().numpy()
            return labels_pred

if __name__ == '__main__':
    pass