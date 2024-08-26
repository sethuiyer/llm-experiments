import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class AttentionLayer(ABC, nn.Module):
    @abstractmethod
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        pass

class DFTMultiHeadAttention(AttentionLayer):
    def __init__(self, embed_dim: int, num_heads: int):
        super(DFTMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        q = self._reshape_for_attention(self.q_proj(query), batch_size)
        k = self._reshape_for_attention(self.k_proj(key), batch_size)
        v = self._reshape_for_attention(self.v_proj(value), batch_size)

        q_dft = torch.fft.fft(q, dim=-1)
        k_dft = torch.fft.fft(k, dim=-1)

        attention_complex = torch.einsum('bhqd,bhkd->bhqk', q_dft, k_dft.conj())
        attention_amplitude = torch.abs(attention_complex)

        if mask is not None:
            attention_amplitude = attention_amplitude.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(attention_amplitude, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(output)

    def _reshape_for_attention(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

class LanguageModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class SimpleLanguageModel(LanguageModel):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            DFTMultiHeadAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = x + layer(x, x, x)
        x = self.layer_norm(x)
        return self.fc(x)

class WorldBuildDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = self.tokenizer.encode(text).ids

    def __len__(self) -> int:
        return len(self.inputs) - self.max_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.inputs[idx:idx+self.max_length]
        mlm_input, mlm_labels = self._prepare_mlm_input(input_ids)
        ntp_input, ntp_label = self._prepare_ntp_input(input_ids)
        return torch.tensor(mlm_input), torch.tensor(mlm_labels), torch.tensor(ntp_input), torch.tensor(ntp_label)

    def _prepare_mlm_input(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        mlm_input = input_ids.copy()
        mlm_labels = [-100] * self.max_length

        for i in range(len(input_ids)):
            if random.random() < 0.15:
                mlm_labels[i] = mlm_input[i]
                if random.random() < 0.8:
                    mlm_input[i] = self.tokenizer.token_to_id("[MASK]")
                elif random.random() < 0.5:
                    mlm_input[i] = random.randint(0, len(self.tokenizer.get_vocab()) - 1)

        return mlm_input, mlm_labels

    def _prepare_ntp_input(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        return input_ids[:-1], input_ids[1:]

class TokenizerFactory:
    @staticmethod
    def create_tokenizer(text: str) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        return tokenizer

class ModelTrainer:
    def __init__(self, model: LanguageModel, optimizer: optim.Optimizer, scheduler: ReduceLROnPlateau, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_and_validate(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> Tuple[List[float], List[float]]:
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader, epoch, num_epochs)
            val_loss = self._validate_epoch(val_loader, epoch, num_epochs)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.scheduler.step(val_loss)

        return train_losses, val_losses

    def _train_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> float:
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            loss = self._process_batch(batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader, epoch: int, num_epochs: int) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                loss = self._process_batch(batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mlm_input, mlm_labels, ntp_input, ntp_labels = [b.to(self.device) for b in batch]

        mlm_output = self.model(mlm_input)
        mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(mlm_output.view(-1, mlm_output.size(-1)), mlm_labels.view(-1))

        ntp_output = self.model(ntp_input)
        ntp_loss = nn.CrossEntropyLoss()(ntp_output.view(-1, ntp_output.size(-1)), ntp_labels.view(-1))

        return mlm_loss + ntp_loss

class Visualizer:
    @staticmethod
    def plot_losses(train_losses: List[float], val_losses: List[float], save_path: str = 'loss_plot.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

class ModelManager:
    @staticmethod
    def save_model(model: LanguageModel, optimizer: optim.Optimizer, train_losses: List[float], val_losses: List[float], path: str):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, path)

    @staticmethod
    def load_model(model: LanguageModel, optimizer: optim.Optimizer, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return {
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses']
        }

class Config:
    def __init__(self):
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        self.max_length = 128
        self.batch_size = 32
        self.num_epochs = 20
        self.learning_rate = 1e-4
        self.val_split = 0.1
        self.weight_decay = 0.01

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("complete_world_build.txt", 'r') as f:
        text = f.read()

    tokenizer = TokenizerFactory.create_tokenizer(text)
    vocab_size = tokenizer.get_vocab_size()

    full_dataset = WorldBuildDataset(text, tokenizer, config.max_length)
    train_size = int((1 - config.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = SimpleLanguageModel(vocab_size, config.embed_dim, config.num_heads, config.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    trainer = ModelTrainer(model, optimizer, scheduler, device)
    train_losses, val_losses = trainer.train_and_validate(train_loader, val_loader, config.num_epochs)

    Visualizer.plot_losses(train_losses, val_losses)
    ModelManager.save_model(model, optimizer, train_losses, val_losses, "enhanced_dft_language_model.pth")

    print("Training complete. Model and training history saved.")

if __name__ == "__main__":
    main()
