from quatfit.training.specialized_training import SpecializedTrainingManager

# Bridge for surprise gate training steps
def train_memory_step(model, hidden_states):
    manager = SpecializedTrainingManager(model, device=next(model.parameters()).device)
    return manager.train_memory_surprise_gate_step(hidden_states)
