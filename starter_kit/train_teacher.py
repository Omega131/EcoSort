import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import argparse

def get_dataloaders(data_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # make sure train and val have exactly the same class folders
    if train_dataset.classes != val_dataset.classes:
        print("ERROR: train and val folders have different class names!")
        print("train:", train_dataset.classes)
        print("val:  ", val_dataset.classes)
        raise SystemExit
        
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Train Teacher Model")
    parser.add_argument('--data_dir', type=str, default='.', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # Default LR lowered to 0.0001 for safer fine-tuning
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate') 
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size) 
    
    # Dynamically grab the number of classes from the folder structure!
    num_classes = len(train_loader.dataset.classes)
    
    print("Downloading pre-trained ResNet-50...")
    teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Replace the final layer using our dynamic num_classes
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    teacher = teacher.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0 # Tracker for the best model
    
    print("Training Teacher Model...")
    for epoch in range(args.epochs):
        teacher.train()
        train_correct, train_total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        scheduler.step()
        train_acc = 100 * train_correct / train_total
        
        # Validation
        teacher.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Only save if validation accuracy strictly improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(teacher.state_dict(), 'teacher.pth')
            print(f"  -> New best validation accuracy! Saved to teacher.pth")

if __name__ == '__main__':
    main()
