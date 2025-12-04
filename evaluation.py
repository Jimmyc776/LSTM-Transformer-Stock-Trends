import torch
from lstm import StockLSTM
from transformer import StockTransformer
from stock_dataloader import create_stock_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_name = 'StockTransformer_ModelMini'

checkpoint = torch.load(f'models/{save_name}.pth', weights_only=True, map_location=device)
model = StockTransformer(
    inp_dim=checkpoint['input_size'],
    d_model=checkpoint['hidden_size'],
    n_heads=4,
    n_layers=checkpoint['num_layers'],
    dim_feedforward=256,
    dropout=checkpoint['dropout'],
    output_dim=1,
    max_len=500
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

total_loss = 0.0
criterion = torch.nn.MSELoss()

with torch.no_grad():
    eval_dataset = create_stock_dataloader(
        stock_csv = 'selected_stocks_data.csv',
        metadata_csv = 'selected_stocks_quality.csv',
        seq_len=100,
        batch_size=32
    )['eval_dataset']
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)

    for batch_x, batch_y in eval_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        total_loss += loss.item() * batch_x.size(0)

    total_loss /= len(eval_loader.dataset)
    print(f"Final Evaluation Loss (MSE): {total_loss:.6f}")