# src/trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os


def train_one_epoch(
    model, loader, optimizer, scheduler,
    device, clip_norm=1.0, accumulation_steps=4
):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Training", leave=False)):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metrics        = batch['metrics'].to(device)
        labels         = batch['label'].to(device)

        # ── Forward pass ─────────────────────────────────────────────────────
        logits = model(input_ids, attention_mask, metrics)

        # ── Sanity check ─────────────────────────────────────────────────────
        if logits is None:
            raise ValueError(
                f"model.forward() returned None at step {step}. "
                "Check model.py forward() method."
            )

        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle remaining steps if dataset not divisible by accumulation_steps
    if (step + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metrics        = batch['metrics'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask, metrics)
            preds  = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return macro_f1, np.array(all_preds), np.array(all_labels)


def train_model(
    model, train_loader, val_loader,
    config, device,
    save_path="checkpoints/best_model.pt"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    accumulation_steps = config.get('accumulation_steps', 4)

    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    total_steps  = (len(train_loader) // accumulation_steps) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1  = 0.0
    history  = []

    print(f"\n  Total steps     : {total_steps}")
    print(f"  Warmup steps    : {warmup_steps}")
    print(f"  Accumulation    : {accumulation_steps}")
    print(f"  Effective batch : {train_loader.batch_size * accumulation_steps}\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, accumulation_steps=accumulation_steps
        )

        val_f1, _, _ = evaluate(model, val_loader, device)

        history.append({
            'epoch':        epoch,
            'train_loss':   round(train_loss, 4),
            'val_macro_f1': round(val_f1, 4),
        })

        print(f"  Loss: {train_loss:.4f}  |  Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best Val Macro F1: {best_f1:.4f}")
    return history