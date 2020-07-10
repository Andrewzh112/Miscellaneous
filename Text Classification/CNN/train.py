import torch
import numpy as np

def train_loop(model, device, epochs, train_loader, 
               test_loader, criterion, optimizer, scheduler=None):
    """typical training loop"""
    train_losses, test_losses = [],[]
    best_test_loss = np.inf

    for epoch in range(1, epochs + 1):
        model.train() 
        batch_losses = []

        for batch in train_loader:
            data = batch['text'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets.type_as(output))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            batch_losses.append(loss.item())
            if scheduler is not None:
                scheduler.step()

        train_losses.append(np.mean(batch_losses))
        
        model.eval()
        batch_losses = []
    
        for batch in test_loader:
            data = batch['text'].to(device)
            targets = batch['targets'].to(device)

            output = model(data)
            loss = criterion(output, targets.type_as(output))
            batch_losses.append(loss.item())

        test_losses.append(np.mean(batch_losses))

        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            torch.save(model, f'{model.model_type}.pt')
        print('epoch {:3d} | loss {:5.8f} | test loss {:5.8f}'.format(epoch,train_losses[-1],test_losses[-1]))
        
    return train_losses, test_losses