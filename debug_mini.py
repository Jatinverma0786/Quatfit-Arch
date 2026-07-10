import sys
import trace
from quatfit.model.config import QuatfitConfig
from quatfit.model.quatfit_model import QuatfitModel

print("Loading config...")
config = QuatfitConfig.get_preset_config('mini')
print("Init QuatfitModel...")
try:
    model = QuatfitModel(config)
except Exception as e:
    import traceback
    traceback.print_exc()
print("Done")
