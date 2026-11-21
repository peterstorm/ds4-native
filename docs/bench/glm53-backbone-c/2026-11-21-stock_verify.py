# Scratch: zai BF16 -> JANGQ FP8 (FP32 amax/448 per 128x128 block, x/s, RNE E4M3) -> kitani recipe == Backbone A,
# on every tensor both CRACK variants leave unedited (o_proj 0-11, shared down_proj 3-19).
import json, sys
sys.path.insert(0, "/w")
import numpy as np, torch
import kitani_recipe as k
wm = json.load(open("/w/bf16-idx.json"))["weight_map"]
k.CYBER_REPO, k.CYBER_REV = "zai-org/GLM-5.3-BF16", "abf42e6c0e53bc29a875511301e7b7fcd220315b"
remote, local = k.RemoteShards(wm), k.LocalShards("/a")
def jangq_fp8(raw, shape):
    R, C = shape
    x = torch.from_numpy(np.frombuffer(raw, np.uint16).astype(np.int16).copy()).view(torch.bfloat16).reshape(R, C).float()
    pr, pc = -R % 128, -C % 128
    xp = torch.nn.functional.pad(x, (0, pc, 0, pr))
    b = xp.reshape((R + pr) // 128, 128, (C + pc) // 128, 128).permute(0, 2, 1, 3)
    s = b.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (b / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    codes = q.permute(0, 2, 1, 3).reshape(R + pr, C + pc)[:R, :C].contiguous().view(torch.uint8).numpy()
    return codes, s[:, :, 0, 0].numpy().astype(np.float32)
names = [f"model.layers.{L}.self_attn.o_proj.weight" for L in range(12)] + \
        [f"model.layers.{L}.mlp.shared_experts.down_proj.weight" for L in range(3, 20)]
bad = 0
for n in names:
    dt, shape, raw = remote.tensor(n)
    for t, (d, sh, out) in k.converted(n, *jangq_fp8(raw, shape)).items():
        want = local.tensor(t)[2]
        same = np.count_nonzero(np.frombuffer(out, np.uint8) == np.frombuffer(want, np.uint8))
        bad += same != len(want)
        print(f"{t}: {same}/{len(want)} bytes identical", flush=True)
print("VERIFIED: bit-exact" if not bad else f"MISMATCH in {bad} tensors", flush=True)
