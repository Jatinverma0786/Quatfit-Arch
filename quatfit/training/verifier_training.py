from quatfit.training.specialized_training import SpecializedTrainingManager

# Bridge for verifier training steps
def train_verifier_step(model, hidden_states, step_labels):
    manager = SpecializedTrainingManager(model, device=next(model.parameters()).device)
    return manager.train_verifier_step(hidden_states, step_labels)
