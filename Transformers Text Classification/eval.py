from sklearn import metrics
import torch

def get_classification_report(data_loader, model, device):
    """Get a sklearn multivariable classification report"""

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            description = batch['description'].to(device)
            brand = batch['brand'].to(device)
            brand_category = batch['brand_category'].to(device)
            name = batch['name'].to(device)
            details = batch['detail'].to(device)
            color = batch['color'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(description,color)

            predictions = torch.sigmoid(outputs).ge(0.5)

            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    return metrics.classification_report(y_true, y_pred, zero_division=True)

def accuracy(data_loader, model, device):
    """Calculate accuracy"""

    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch in data_loader:
            description = batch['description'].to(device)
            brand = batch['brand'].to(device)
            brand_category = batch['brand_category'].to(device)
            name = batch['name'].to(device)
            details = batch['detail'].to(device)
            color = batch['color'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(description,color)

            predictions = torch.sigmoid(outputs).ge(0.5)

            n_correct += (predictions == targets).sum().item()
            n_total += targets.size(0)*targets.size(1)

    return n_correct / n_total