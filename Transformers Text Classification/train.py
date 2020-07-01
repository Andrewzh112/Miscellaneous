import torch
import numpy as np

def train_loop(model, device, epochs, train_loader, test_loader, criterion, optimizer, scheduler):
    for epoch in range(1, epochs + 1):
        model.train() 
        train_losses, test_losses = [],[]
        best_test_loss = np.inf
        for batch in train_loader:
            batch_losses = []
            data = batch['input'].to(device)
            colors = batch['embed'].to(device)
            targets = batch['targets'].to(device)
            optimizer.zero_grad()
            output = model(src=data, 
                           colors=colors)
            loss = criterion(output, targets.type_as(output))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))
        scheduler.step()
        
        model.eval()
        for batch in test_loader:
            batch_losses = []
            data = batch['input'].to(device)
            targets = batch['targets'].to(device)
            colors = batch['embed'].to(device)
            output = model(src=data, 
                           colors=colors)
            loss = criterion(output, targets.type_as(output))
            batch_losses.append(loss.item())
        test_losses.append(np.mean(batch_losses))
        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            torch.save(model, 'transformer.pt')
        print('epoch {:3d} | loss {:5.8f} | test loss {:5.8f}'.format(epoch,train_losses[-1],test_losses[-1]))
    return train_losses, test_losses