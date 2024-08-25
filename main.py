import torch
import os
from timeit import default_timer as timer

from src.data_setup import data_setup
from src.model import create_effnetb2_model, get_transforms
from src.train_and_test import train

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dir = "data/train"
    test_dir = "data/test"

    transforms = get_transforms()
    train_dataloader, test_dataloader, class_names = data_setup(
        train_dir, test_dir, transforms, batch_size=32, num_workers=os.cpu_count()
    )

    model = create_effnetb2_model(num_classes=len(class_names)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start_time = timer()
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=25,
        device=device
    )
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

if __name__ == "__main__":
    main()