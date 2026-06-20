import torch
import sentencepiece as spm
from model import TinyGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer_types = ["bpe", "unigram", "word", "char"]
block_size = 64 
embedding_dim = 64
n_heads = 4  
n_layers = 4 
batch_size = 32

print(f"{'='*50}")
print("Evaluasi Final Loss (Cross-Entropy)")
print("Catatan: LLM umumnya menggunakan metrik Loss/Perplexity")
print("bukan Akurasi seperti pada klasifikasi gambar.")
print(f"{'='*50}\n")

for t_type in tokenizer_types:
    model_prefix = f"tokenizer_{t_type}"
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        vocab_size = sp.get_piece_size()
        ids = sp.encode(text, out_type=int) 
        data = torch.tensor(ids, dtype=torch.long) 
        
        model = TinyGPT(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim, 
            block_size=block_size, 
            n_heads=n_heads, 
            n_layers=n_layers
        ).to(device)
        
        model.load_state_dict(torch.load(f"tinygpt_{t_type}.pt", map_location=device))
        model.eval()

        # Ambil sampel batch secara acak untuk menghitung estimasi Loss akhir
        ix = torch.randint(len(data) - block_size, (batch_size,))  
        x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
        
        with torch.no_grad():
            logits, loss = model(x, y)
            
        print(f"[{t_type.upper():7s}] Final Estimated Loss : {loss.item():.4f}")
        
    except Exception as e:
        print(f"[{t_type.upper():7s}] Error mengevaluasi: {e}")

print(f"\n{'='*50}")
