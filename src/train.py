import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch

# Import models
from models.baseline_cnn import BaselineCNN
from models.transfer_learning import get_resnet50, get_vit


# =====================================================
# Hyperparameters & Paths
# =====================================================
DATA_DIR = "data/processed"
SAVE_DIR = "models/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# Training Function
# =====================================================
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device, model_name):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ---------- MLflow logging ----------
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        # ---------- Save best model ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_best.pth"))
            mlflow.pytorch.log_model(model, artifact_path=f"{model_name}_model")

    return model


# =====================================================
# Main
# =====================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_test_transforms = train_transforms

    # Datasets & Dataloaders
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)

    # ---------- Model Selection ----------
    if args.model == "baseline":
        model = BaselineCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif args.model == "resnet50":
        model, criterion, optimizer = get_resnet50(num_classes, lr=args.lr, device=device)

    elif args.model == "vit":
        model, criterion, optimizer = get_vit(num_classes, lr=args.lr, device=device)

    else:
        raise ValueError("Invalid model choice! Use: baseline | resnet50 | vit")

    # ---------- MLflow ----------
    mlflow.set_experiment("Chest_XRay_Models")
    with mlflow.start_run(run_name=f"{args.model}_finetune"):
        mlflow.log_params({
            "model": args.model,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "img_size": args.img_size
        })

        # Train & validate
        train_model(model, criterion, optimizer, train_loader, val_loader,
                    args.epochs, device, args.model)


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", help="baseline | resnet50 | vit")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)
