import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import sentencepiece as spm
from model import TinyGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_types = ["bpe", "unigram", "word", "char"]
block_size = 64 
embedding_dim = 64
n_heads = 4  
n_layers = 4 

prompt_text = "Matahari adalah pusat"

print(f"{'='*60}")
print(f"Prompt: '{prompt_text}'")
print(f"{'='*60}")

for t_type in tokenizer_types:
    model_prefix = f"tokenizer_{t_type}"
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        
        vocab_size = sp.get_piece_size()
        
        model = TinyGPT(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim, 
            block_size=block_size, 
            n_heads=n_heads, 
            n_layers=n_layers
        ).to(device)
        
        model.load_state_dict(torch.load(f"tinygpt_{t_type}.pt", map_location=device))
        model.eval()

        # Encode prompt
        context = torch.tensor([sp.encode(prompt_text, out_type=int)], dtype=torch.long).to(device)
        
        # Generate text
        out = model.generate(context, max_new_tokens=50)
        generated_ids = out[0].tolist()
        generated_text = sp.decode(generated_ids)
        
        print(f"\n[{t_type.upper()}] Generated text:\n")
        print(generated_text)
        print("-" * 60)
        
    except Exception as e:
        print(f"\n[{t_type.upper()}] Could not load model or generate text. Error: {e}")

