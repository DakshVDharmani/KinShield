import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda', model_path='rppg_model.pth'):
    # Calculate class weights for imbalanced data
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    pos_count = (all_labels == 1).sum()
    neg_count = (all_labels == 0).sum()
    pos_weight = neg_count / (pos_count + 1e-6)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    model.to(device)
    best_auc = 0.0
    best_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            with torch.no_grad():
                preds = (outputs > 0.5).float()
                train_preds.extend(outputs.detach().cpu().numpy().flatten())
                train_labels.extend(labels.cpu().numpy().flatten())
            
            pbar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_preds_arr = np.array(train_preds)
        train_labels_arr = np.array(train_labels)
        train_auc = roc_auc_score(train_labels_arr, train_preds_arr) if len(set(train_labels_arr)) > 1 else 0.5
        train_acc = np.mean((train_preds_arr > 0.5) == train_labels_arr)
        
        # Validation
        val_loss, val_acc, val_auc, val_preds, val_labels = validate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"[BEST] Saved better model with AUC: {best_auc:.4f}, Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining Complete! Best AUC: {best_auc:.4f}, Best Acc: {best_acc:.4f}")
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_labels.extend(labels.cpu().numpy().flatten())
    
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    
    val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
    val_acc = np.mean((val_preds > 0.5) == val_labels)
    val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(val_labels, val_preds > 0.5).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return val_loss, val_acc, val_auc, val_preds, val_labels
