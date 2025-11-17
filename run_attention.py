import torch
from src.transformer.utils import get_bert_model, embed_sentence, plot_attention_heads, load_bert_weights_into_mha
from src.transformer.attention import MultiHeadAttention

def test_mha(layer_id: int):
    my_mha = load_bert_weights_into_mha(bert_model, layer_id)
    my_mha.eval()
    
    with torch.no_grad():
        _, my_attn = my_mha(embeddings, mask)
        plot_attention_heads(my_attn, tokens, f"my_mHA_layer_{layer_id}_attention.png")

if __name__ == "__main__":
    sentence = "If I had a million dollars, I'd spend it all."
    
    tokenizer, bert_model = get_bert_model()
    embeddings, mask, tokens = embed_sentence(sentence, tokenizer, bert_model)
    
    with torch.no_grad():
        # All the attention layers
        outputs = bert_model(**tokenizer(sentence, return_tensors="pt"), output_attentions=True)
        # First layer
        bert_attn = outputs.attentions[0]
        plot_attention_heads(bert_attn, tokens, "bert_layer_0_attention.png")
    
    # Test own implementation
    LAYER_IDS = [0, 11]
    for layer_id in LAYER_IDS:
        test_mha(layer_id=layer_id)