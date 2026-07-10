import torch
from quatfit.model.config import QuatfitConfig
from quatfit.model.quatfit_model import QuatfitModel
from quatfit.serving.paged_cache import PagedKVCacheManager

def test_preset(preset_name):
    print(f"\n--- Testing preset: {preset_name} ---")
    config = QuatfitConfig.get_preset_config(preset_name)
    
    # Downscale dimensions to avoid OOM in testing
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_dense_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_groups = 2
    config.head_dim = 16
    config.expert_ffn_dim = 64
    if config.dense_ffn_dim is not None:
        config.dense_ffn_dim = 64
    if config.num_experts > 2:
        config.num_experts = 2
    if config.top_k > config.num_experts:
        config.top_k = config.num_experts
    config.vocab_size = 1000
    config.adaptive_exit_checkpoints = [1]
    
    model = QuatfitModel(config)
    
    # 1. Forward Pass Test
    print("Running forward pass...")
    bsz, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len), dtype=torch.long)
    outputs = model(input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (bsz, seq_len, config.vocab_size)
    print("Forward pass OK.")
    
    # 2. Generate Pass Test
    print("Running generate pass...")
    gen_ids = torch.randint(0, config.vocab_size, (1, 4), dtype=torch.long)
    gen_outputs = model.generate(gen_ids, max_new_tokens=3)
    assert gen_outputs.shape == (1, 7)
    print("Generate pass OK.")
    
    # 3. PagedKV Cache Test (if attention supports it)
    print("Running PagedKVCache forward pass...")
    caches = []
    for _ in range(config.num_hidden_layers):
        c = PagedKVCacheManager(
            num_blocks=10, 
            num_heads=getattr(config, 'num_key_value_groups', config.num_attention_heads), 
            block_size=4, 
            head_dim=config.head_dim,
            config=config
        )
        # Pre-allocate blocks for batch size 1
        c.context_lengths = [0]
        c.block_tables = [c.allocate(seq_len=8)]
        caches.append(c)
    
    paged_input = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)
    paged_out = model(paged_input, past_key_values=caches, use_cache=True)
    assert paged_out["past_key_values"][0] is caches[0]
    assert caches[0].context_lengths[0] == 8
    print("PagedKVCache pass OK.")
    print(f"{preset_name} PASSED.")

if __name__ == "__main__":
    presets = ["nano", "mini", "base", "pro", "ultra"]
    for preset in presets:
        try:
            test_preset(preset)
        except Exception as e:
            print(f"FAILED on {preset}: {e}")
            import traceback
            traceback.print_exc()
