import torch
from MedIQA_functions import get_data_loader, test


if __name__ == '__main__':
    # Configuration
    model_path = ""
    batch_size = 1
    split = ''

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.load(model_path).to(device)
    model.eval()

    # Data loader
    test_loader = get_data_loader(random=False, batch_size=batch_size, split=split, model=model)

    # Run test
    test_loss, SRCC_test, PLCC_test, RMSE_test = test(model, test_loader, device)

    # Print results
    print("=" * 50)
    print("Test Results")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test SRCC: {SRCC_test:.4f}")
    print(f"Test PLCC: {PLCC_test:.4f}")
    print(f"Test RMSE: {RMSE_test:.4f}")
    print("=" * 50)
