import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms

# Import our custom EcoNet
from model import EcoNet

def get_student_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Dataloader specifically for the student model.
    Uses milder augmentations than the teacher because the student has far less capacity.
    """
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # Mild rotation instead of heavy TrivialAugment
        transforms.ToTensor(),
        # No RandomErasing for the tiny student!
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    
    # drop_last=True prevents noisy batchnorm stats on tiny final batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader

def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # F.cross_entropy is cleaner than rebuilding nn.CrossEntropyLoss
    ce_loss = F.cross_entropy(student_logits, labels)
    
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs) * (T * T)
    
    return alpha * ce_loss + (1. - alpha) * kd_loss

def validate_model(model, val_loader, device, model_name="Model"):
    """Helper function to validate a model and return accuracy"""
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    print(f"[{model_name}] Validation Accuracy: {val_acc:.2f}%")
    return val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument('--data_dir', type=str, default='.', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train student')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--teacher_weights', type=str, default='teacher.pth', help='Path to teacher weights')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation Temperature (T)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation Alpha (weight for CE loss)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    train_loader, val_loader = get_student_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    num_classes = len(train_loader.dataset.classes)
    
    # 2. Setup Teacher
    print(f"Loading Teacher Model from {args.teacher_weights}...")
    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    
    if not os.path.exists(args.teacher_weights):
        print(f"ERROR: Teacher weights not found at {args.teacher_weights}")
        return
        
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))
    teacher = teacher.to(device)
    
    # Check teacher accuracy before distillation
    print("\nRunning sanity check on the loaded Teacher...")
    validate_model(teacher, val_loader, device, model_name="Teacher")
    
    teacher.eval() # Do not train the teacher
    
    # 3. Setup Student
    print("\nInitializing Student Model (EcoNet)...")
    student = EcoNet(num_classes=num_classes)
    student = student.to(device)
    
    # 4. Optimization
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    student_weights_path = os.path.join(args.data_dir, 'student.pth')
    
    print(f"\nStarting Knowledge Distillation (T={args.temperature}, Alpha={args.alpha})...")
    for epoch in range(args.epochs):
        student.train()
        train_correct, train_total = 0, 0
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
                
            loss = distillation_loss(student_logits, teacher_logits, labels, T=args.temperature, alpha=args.alpha)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        scheduler.step()
        
        train_acc = 100 * train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        val_acc = validate_model(student, val_loader, device, model_name=f"Student Epoch {epoch+1:02d}/{args.epochs}")
        print(f"  -> Train Acc: {train_acc:.2f}% | Loss: {avg_loss:.4f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), student_weights_path)
            print(f"  [!] New best student validation accuracy! Saved to {student_weights_path}\n")

    print("Knowledge Distillation Complete!")
    
    # 5. Export to ONNX
    print("\n--- Exporting Best Model to ONNX ---")
    
    # Reload the very best weights
    student.load_state_dict(torch.load(student_weights_path, map_location=device))
    student.eval()
    student.to('cpu') # Exporting on CPU is safer and more standard
    
    onnx_path = os.path.join(args.data_dir, 'model.onnx')
    dummy_input = torch.randn(1, 3, 96, 96)
    
    try:
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            student,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )
    
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX Export Successful! Saved to {onnx_path}")
    print(f"File Size: {size_mb:.2f} MB")
    if size_mb > 1.5:
        print("WARNING: Model exceeds 1.50 MB limit!")
    else:
        print("SUCCESS: Model is within the 1.50 MB limit!")

if __name__ == '__main__':
    main()
