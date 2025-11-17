import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .attention import MultiHeadAttention

def get_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased")
    return tokenizer, bert

def embed_sentence(text: str, tokenizer, bert_model):
    # Turn a string/sentence into a vector
    token_ids = tokenizer(text, return_tensors="pt") # A dictionary
    embeddings = bert_model.embeddings(token_ids["input_ids"]) # Just mapped each of those integers into a vector
    mask = token_ids["attention_mask"] # B x L
    mask = mask.unsqueeze(1).unsqueeze(2) # B x 1 x 1 x L so that it can be broadcast to B x H x L x L
    
    token_strings = tokenizer.convert_ids_to_tokens(token_ids["input_ids"][0])
    return embeddings, mask, token_strings

def plot_attention_heads(attn: torch.tensor, tokens: torch.tensor, filename: str, heads_per_row:int=4):
    attn = attn[0].detach().cpu()
    H, _, _ = attn.shape
    rows = (H + heads_per_row - 1)//heads_per_row
    _, axes = plt.subplots(rows, heads_per_row, figsize=(4*heads_per_row,4*rows))
    axes = axes.flatten()
    for h in range(H):
        sns.heatmap(attn[h], vmin=0, vmax=attn.max(),
                    cmap="viridis", square=True,
                    xticklabels=tokens, yticklabels=tokens,
                    ax=axes[h], cbar=False)
        axes[h].set_title(f"Head {h}")
        axes[h].tick_params(labelsize=8, rotation=90)
    for ax in axes[H:]:
        ax.axis("off")
    plt.tight_layout()
    Path("src/transformer/outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"outputs/{filename}")
    
def load_bert_weights_into_mha(bert_model, layer_id: int) -> MultiHeadAttention:
    # Loads the corresponding pre-trained attention head from the given layer in the bert model into a new Attention Head that we return
    model = MultiHeadAttention(d_model=768, num_heads=12)
    bert_layer = bert_model.encoder.layer[layer_id]
    # Attention parts
    bert_sa = bert_layer.attention.self
    bert_output = bert_layer.attention.output
    
    # Load weights
    with torch.no_grad():
        model.W_q.weight.data.copy_(bert_sa.query.weight)
        model.W_q.bias.data.copy_(bert_sa.query.bias)
        
        model.W_k.weight.data.copy_(bert_sa.key.weight)
        model.W_k.bias.data.copy_(bert_sa.key.bias)
        
        model.W_v.weight.data.copy_(bert_sa.value.weight)
        model.W_v.bias.data.copy_(bert_sa.value.bias)
        
        model.W_o.weight.data.copy_(bert_output.dense.weight)
        model.W_o.bias.data.copy_(bert_output.dense.bias)
    
    return model