import imp
import os
import time
import logging
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multimodal model training script')
    parser.add_argument('--data-root', type=str, default=r'maturity5000_PCA40_Reduced_3D', 

                      help='Dataset root directory')
    parser.add_argument('--save-dir', type=str, default='Result/maturity', 
                      help='Model and results save directory')

    #parser.add_argument('--class-names', type=str, required=True, help='Class names, separated by commas')    #（optional）

    parser.add_argument('--batch-size', type=int, default=32, 
                      help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, 
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, 
                      help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=5, 
                      help='Early stopping patience')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    
    # Training configuration
    config = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'save_dir': args.save_dir,
        'data_root': args.data_root,
        'patience': args.patience,
        #'class_names': list(args.class_names.split(',')),
        #'class_names': ['2-4', '0-2', '6-8', '8-10', '4-6']    #color
        'class_names': ['6-8', '4-6', '8-10', '0-4']  #maturity
        #'class_names': ['5-8', '8-10', '3-5', '0-3']   #structure
        #'class_names': ['7-10', '4-7', '0-4'] #identity
        #'class_names': ['8-10', '0-3', '5-8', '3-5']    #oil_content
        #'class_names': ['4-6', '2-4', '6-8', '8-10', '0-2']  #chroma

    }
    
    # Initialize environment
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize dataset and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try to import custom dataset class
    try:
        # Assume MyDataset is defined in dataset module
        from my_dataset import MyDataset
    except ImportError:
        logging.error("Failed to import MyDataset, please ensure the custom dataset class is properly defined")
        return
    
    # Load pre-split training, validation, and test sets
    try:
        train_dataset = MyDataset(
            root_dir=os.path.join(config['data_root'], 'train'), 
            transform=transform,augment=True,
        )
        val_dataset = MyDataset(
            root_dir=os.path.join(config['data_root'], 'val'), 
            transform=transform,augment=False,
        )
        test_dataset = MyDataset(
            root_dir=os.path.join(config['data_root'], 'test'), 
            transform=transform,
        )
        
        logging.info(f"Dataset loading completed - Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}")
    except Exception as e:
        logging.error(f"Dataset loading failed: {str(e)}")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    try:
        # Assume MultiModalModel is defined in model module
        from WaveMF import WaveMF
        model = WaveMF(num_classes=len(config['class_names'])).to(device)
        # from SSFTT import SSFTTnet
        # model = SSFTTnet(num_classes=len(config['class_names'])).to(device)
        # from morphformer import CNN
        # model = CNN(16, 25, len(config['class_names']), False).to(device)
        # from deformer import  DSFormer
        # model = DSFormer(dataset='ip', kernel_size=3,  ps=2, k=0.8, group_num=4, emb_dim=128, num_classes=len(config['class_names'])).to(device)
        # from FDGC import FDGC
        # model = FDGC(input_channels=25, num_nodes=len(config['class_names']), num_classes=len(config['class_names']), patch_size=16).to(device)



        logging.info(f"Model initialization completed - Number of classes: {len(config['class_names'])}")
    except ImportError:
        logging.error("Failed to import MultiModalTobaccoClassifier, please ensure the model class is properly defined")
        return
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    # Optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,  
        weight_decay=5e-2,   # Weight decay
        betas=(0.9, 0.999)
    )
    
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    # Define warmup scheduler, increasing warmup epochs from 1 to 5 for smoother learning rate increase
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=2)
    # Let cosine annealing run for the remaining training epochs, while adjusting the minimum learning rate
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'] - 2, eta_min=1e-7)
    # Use SequentialLR to chain the two schedulers
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[2]  # Switch to cosine annealing scheduler after epoch 10
    )
    
    # Training records
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_acc = 0.0
    no_improve_epochs = 0  # Early stopping counter

    # Training loop
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for hsi, rgb, labels in train_pbar:
            hsi = hsi.to(device)
            rgb = rgb.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(hsi, rgb)
            
            ce = criterion(outputs, labels)
            #Calculate wavelet loss
            loss_wavelet = model.get_wavelet_loss()
            #Total loss
            loss = ce + loss_wavelet 
            #loss = ce 



            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar information
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Validation set progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
        with torch.no_grad():
            for hsi, rgb, labels in val_pbar:
                hsi = hsi.to(device)
                rgb = rgb.to(device)
                labels = labels.to(device)
                
                outputs = model(hsi, rgb)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update validation progress bar information
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            logging.info(f"Saved best model, accuracy: {val_acc:.4f}")
            no_improve_epochs = 0  # Reset early stopping counter
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config['patience']:
                logging.info(f"Early stopping triggered: {config['patience']} epochs without improvement")
                break
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history
        }, os.path.join(config['save_dir'], 'last_checkpoint.pth'))
        
        # Print log
        epoch_time = time.time() - start_time
        logging.info(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Time: {epoch_time:.1f}s | "
        f"Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f} | "
        f"Training accuracy: {train_acc:.4f} | Validation accuracy: {val_acc:.4f} | "
        f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Update learning rate
        scheduler.step()
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best_model.pth'), weights_only=True))
    model.eval()
    
    # Testing phase, collect all predictions and labels for detailed evaluation
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for hsi, rgb, labels in test_loader:
            hsi = hsi.to(device)
            rgb = rgb.to(device)
            labels = labels.to(device)
            
            outputs = model(hsi, rgb)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    
    # Calculate OA, AA, KPA
    cm = confusion_matrix(all_labels, all_preds)
    oa = np.trace(cm) / np.sum(cm)
    aa = np.mean(cm.diagonal() / cm.sum(axis=1))
    kpa = cohen_kappa_score(all_labels, all_preds)

    logging.info(f"Test loss: {test_loss:.4f} | Test OA: {oa:.4f} | Test AA: {aa:.4f} | Test KPA: {kpa:.4f}")
    
    # Generate and save classification report
    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=config['class_names'],
        digits=4
    )
    logging.info(f"\nClassification Report:\n{class_report}")
    
    with open(os.path.join(config['save_dir'], 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
        f.write(f"Average Accuracy (AA): {aa:.4f}\n")
        f.write(f"Kappa Coefficient (KPA): {kpa:.4f}\n\n")
        f.write(class_report)
    
    # Generate and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=config['class_names'],
        yticklabels=config['class_names']
    )
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'confusion_matrix.png'))
    plt.close()
    
    # Save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training accuracy')
    plt.plot(history['val_acc'], label='Validation accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.close()

if __name__ == "__main__":
    main()