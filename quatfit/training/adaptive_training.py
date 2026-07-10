from quatfit.training.specialized_training import SpecializedTrainingManager

# Bridge for adaptive compute calibration
def calibrate_thresholds(model, eval_dataloader, target_accuracy_drop=0.01):
    manager = SpecializedTrainingManager(model, device=next(model.parameters()).device)
    manager.calibrate_adaptive_thresholds(eval_dataloader, target_accuracy_drop)
