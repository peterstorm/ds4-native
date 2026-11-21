# Scratch: kitani_recipe.verify against zai-org/GLM-5.3 (the stock FP8) instead of the cyber release.
import json, sys
sys.path.insert(0, sys.argv[1])
import kitani_recipe as k
k.CYBER_REPO, k.CYBER_REV = "zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42"
names = [f"model.layers.{L}.self_attn.o_proj.weight" for L in range(12)] + \
        [f"model.layers.{L}.mlp.shared_experts.down_proj.weight" for L in range(3, 20)]
wm = json.load(open(sys.argv[2]))["weight_map"]
sys.exit(1 if k.verify(sys.argv[3], names, wm) else 0)
