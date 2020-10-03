import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(train_dataloader:DataLoader, validation_dataloader:DataLoader, epochs, model:nn.Module, optimizer, scheduler, criterion):
    train_losses = []
    validation_losses = []

    #train-validation loop
    for epoch in range(epochs):
        batch_losses = []
        #training loop
        for _idx , data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        training_loss = np.mean(batch_losses)
        train_losses.append(training_loss)
        scheduler.step()

        #validation loop
        with torch.no_grad():
            val_losses = []
            for _idx, data in enumerate(validation_dataloader):
                inputs, labels = data
                model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        print(f"[{epoch+1}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

    return model, train_losses, validation_losses

def eval_model(test_dataloader: DataLoader, model: nn.Module, criterion):
    test_losses = []
    with torch.no_grad():
        for data in enumerate(test_dataloader):
            inputs, labels = data
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
        test_loss = np.mean(test_losses)
        print(f"Final test loss: {test_loss:.4f}")    
    return test_loss

