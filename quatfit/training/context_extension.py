from quatfit.training.specialized_training import SpecializedTrainingManager

# Bridge for YaRN scaling updates
def extend_context_window(model, target_context_len):
    manager = SpecializedTrainingManager(model, device=next(model.parameters()).device)
    manager.apply_yarn_scaling(target_context_len)
