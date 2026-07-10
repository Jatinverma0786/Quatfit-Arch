import os
import torch
import torch.nn.functional as F
from quatfit.model.config import QuatfitConfig
from quatfit.model.quatfit_model import QuatfitModel
from quatfit.training.loss import QuatfitLoss
from quatfit.training.chat_dataloader import ChatStreamingDataLoader
from quatfit.tokenizer import SimpleBPETokenizer

def prepare_tokenizer(path="chat_tokenizer.json"):
    if not os.path.exists(path):
        print("Training fresh chat tokenizer...")
        tokenizer = SimpleBPETokenizer(vocab_size=2000) # Small vocab for fast test
        # A tiny corpus to ensure we can encode basic chat structures
        corpus = "hello world how are you I am fine thank you the quick brown fox jumps over the lazy dog <|user|> <|assistant|> " * 10
        tokenizer.train(corpus)
        tokenizer.save(path)
    return path

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer_path = prepare_tokenizer()
    tokenizer = SimpleBPETokenizer.load(tokenizer_path)
    
    # Configure model (scaled down for rapid testing)
    config = QuatfitConfig.get_preset_config("nano")
    config.vocab_size = tokenizer.vocab_size
    config.hidden_size = 256
    config.num_hidden_layers = 4
    config.num_dense_layers = 2
    config.num_attention_heads = 4
    config.is_moe = True
    config.num_experts = 8
    config.top_k = 2
    config.sliding_window_size = 128
    
    model = QuatfitModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataloader = ChatStreamingDataLoader(
        dataset_name="WithinUsAI/claude_mythos_distilled_25k",
        split="train",
        max_seq_len=128,
        batch_size=2,
        tokenizer_path=tokenizer_path
    )
    
    model.train()
    print("Starting chat training loop...")
    
    steps = 10
    for step in range(steps):
        try:
            inputs, targets, mask = dataloader.get_batch(device=device)
        except StopIteration:
            break
            
        optimizer.zero_grad()
        
        outputs = model(inputs)
        logits = outputs["logits"] # [batch, seq, vocab]
        
        # Calculate loss using QuatfitLoss to properly handle MTP and masking
        loss_fn = QuatfitLoss(mtp_loss_weight=0.0, exit_loss_weight=0.0) # simplify for chat script
        
        # We need to shift everything correctly for QuatfitLoss
        # It expects unshifted logits and targets, and internally shifts them.
        # Wait, QuatfitLoss expects outputs dict and targets.
        
        final_loss, _ = loss_fn(outputs, targets, loss_mask=mask)
        
        final_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}/{steps} - Loss: {final_loss.item():.4f}")
        
    print("Training complete. Testing chat generation...")
    model.eval()
    
    input_ids = torch.tensor([[
        tokenizer.encoder.get("<|user|>", 4), 
        *tokenizer.encode("hello"), 
        tokenizer.encoder.get("<|assistant|>", 5)
    ]], device=device)
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(input_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            
    response = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated: {response}")

if __name__ == "__main__":
    train()
