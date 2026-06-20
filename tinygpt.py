import torch
import sentencepiece as spm
import os
from model import TinyGPT

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer types
tokenizer_types = ["bpe", "unigram", "word", "char"]
block_size = 64 
embedding_dim = 64
n_heads = 4  
n_layers = 4 
lr = 1e-3 
epochs = 1500  
batch_size = 32

for t_type in tokenizer_types:
    print(f"\n{'='*40}")
    print(f"Training Tokenizer & Model: {t_type.upper()}")
    print(f"{'='*40}")

    vocab_size = 150
    if t_type == "char":
        vocab_size = 60

    model_prefix = f"tokenizer_{t_type}"
    
    # Train SentencePiece
    try:
        spm.SentencePieceTrainer.Train(
            input="corpus.txt",
            model_prefix=model_prefix,
            vocab_size=vocab_size,     
            model_type=t_type,
            character_coverage=0.9995 if t_type != "char" else 1.0
        )
    except Exception as e:
        print(f"Error training tokenizer {t_type}: {e}")
        # fallback to a smaller vocab if Word/Char fails
        vocab_size = 100
        spm.SentencePieceTrainer.Train(
            input="corpus.txt",
            model_prefix=model_prefix,
            vocab_size=vocab_size,     
            model_type=t_type
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    
    ids = sp.encode(text, out_type=int) 
    data = torch.tensor(ids, dtype=torch.long) 
    actual_vocab_size = sp.get_piece_size()
    print(f"Vocab size for {t_type}: {actual_vocab_size}")
    print(f"Total tokens in data: {len(data)}")

    def get_batch():
        ix = torch.randint(len(data) - block_size, (batch_size,))  
        x = torch.stack([data[i:i+block_size] for i in ix])  
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
        return x.to(device), y.to(device)

    # Init model
    model = TinyGPT(
        vocab_size=actual_vocab_size, 
        embedding_dim=embedding_dim, 
        block_size=block_size, 
        n_heads=n_heads, 
        n_layers=n_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(epochs):
        xb, yb = get_batch() 
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0 or step == epochs - 1:
            print(f"[{t_type.upper()}] Step {step}, loss={loss.item():.4f}")

    # Save model weights
    torch.save(model.state_dict(), f"tinygpt_{t_type}.pt")
    print(f"Saved tinygpt_{t_type}.pt")

print("\nAll models trained successfully!")