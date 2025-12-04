import torch
from lstm import StockLSTM
from transformer import StockTransformer
from stock_dataloader import create_stock_dataloader

def evaluate_model(model: torch.nn.Module, eval_loader: torch.utils.data.DataLoader, device: str='cpu') -> float:
    """
    Evaluate the trained model on evaluation dataset
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)

    total_loss /= len(eval_loader.dataset)
    return total_loss