import torch
import os
from lstm import StockLSTM
from transformer import StockTransformer
from stock_dataloader import create_stock_dataloader
from datetime import datetime

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """
    Train LSTM/Transformer model on stock data
    """

    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ### JUSTIFY ###
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return model

def save_model(model, save_name, save_dir='models'):
    """
    Save trained omdel + optimizer + metadata
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f"{save_name}_{timestamp}.pth")

    # Save everything needed for evaluation
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
        'timestamp': timestamp,
        'input_size': getattr(model, 'input_size', 1),
        'hidden_size': getattr(model, 'hidden_size', 64),
        'num_layers': getattr(model, 'num_layers', 2),
        'dropout': getattr(model, 'dropout', 0.2),
        'seq_len': 100,  # From dataloader
        'architecture': 'StockLSTM' if isinstance(model, StockLSTM) else 'StockTransformer'
    })

    print(f"✅ Model saved: {save_path}")
    return save_path


if __name__ == "__main__":
    # Example usage
    SEQ_LEN = 100
    BATCH_SIZE = 32
    INPUT_SIZE = 1
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROP_OUT = 0.2
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cpu'

    stock_csv = 'selected_stocks_data.csv'  # Pre-downloaded stock prices
    metadata_csv = 'selected_stocks_quality.csv'  # Metadata with categories and qualities

    train_loader = create_stock_dataloader(stock_csv, metadata_csv, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)['train_loader']

    model = StockLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, drop_out=DROP_OUT)
    # model = StockTransformer()
    trained_model = train_model(model, train_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
    save_model(trained_model, save_name='StockLSTM_Model')
