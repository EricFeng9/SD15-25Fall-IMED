import inspect
from diffusers import LoRAAttnProcessor, LoRAAttnProcessor2_0
print('LoRAAttnProcessor.__init__ signature:', inspect.signature(LoRAAttnProcessor.__init__))
print('LoRAAttnProcessor2_0.__init__ signature:', inspect.signature(LoRAAttnProcessor2_0.__init__))
