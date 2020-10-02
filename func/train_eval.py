import torch
import numpy as np

def train_model(train_data, validation_data, epochs, model, optimizer, scheduler, criterion):
    train_losses = []
    validation_losses = []

    #train-validation loop
    for epoch in range(epochs):
        batch_losses = []
        #training loop
        for _idx , data in enumerate(train_data):
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
            for _idx, data in enumerate(validation_data):
                inputs, labels = data
                model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

        print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

    return model, train_losses, validation_losses

def eval_model(test_data, model, criterion):
    test_losses = []
    with torch.no_grad():
        for i , data in enumerate(test_data,0):
            inputs, labels = data
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            if (i + 1) % 50 ==0:
                print(f"Test Loss for batch {i+1} : {loss.item():.3f}")
            test_losses.append(loss.item())
    return test_losses

