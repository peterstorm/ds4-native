# Scratch: which FP8 quantizer turns zai's BF16 into the FP8 kitaniai recovered (Backbone A, unedited o_proj)?
import json, sys
sys.path.insert(0, "/w")
import numpy as np, torch
import kitani_recipe as k
name = sys.argv[1] if len(sys.argv) > 1 else "model.layers.0.self_attn.o_proj.weight"
wm = json.load(open("/w/bf16-idx.json"))["weight_map"]
k.CYBER_REPO, k.CYBER_REV = "zai-org/GLM-5.3-BF16", "abf42e6c0e53bc29a875511301e7b7fcd220315b"
dt, shape, raw = k.RemoteShards(wm).tensor(name)
x = torch.from_numpy(np.frombuffer(raw, np.uint16).astype(np.int16).copy()).view(torch.bfloat16).reshape(shape)
want = np.frombuffer(k.LocalShards("/a").tensor(name)[2], np.uint16).reshape(shape)
R, C = shape
def blocks(t):  # [R, C] -> [R/128, C/128, 128, 128]
    return t.reshape(R // 128, 128, C // 128, 128).permute(0, 2, 1, 3)
def unblocks(t):
    return t.permute(0, 2, 1, 3).reshape(R, C)
def quant(x32, scale_of, div=True):
    b = blocks(x32)
    amax = b.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12)
    s = scale_of(amax)
    q = (b / s if div else b * (1.0 / s)).clamp(-448, 448).to(torch.float8_e4m3fn)
    rec = q.to(torch.float32) * s
    return k.bf16_bits(unblocks(rec).numpy())
variants = {
  "fp32 amax/448, x/s": lambda: quant(x.float(), lambda a: a / 448.0),
  "fp32 amax/448, x*(1/s)": lambda: quant(x.float(), lambda a: a / 448.0, div=False),
  "bf16 scale": lambda: quant(x.float(), lambda a: (a / 448.0).to(torch.bfloat16).float()),
  "pow2 scale (ceil)": lambda: quant(x.float(), lambda a: torch.exp2(torch.ceil(torch.log2(a / 448.0)))),
  "bf16 input as-is": lambda: np.frombuffer(raw, np.uint16).reshape(shape),
}
for v, f in variants.items():
    got = f()
    print(f"{v:28s} {np.count_nonzero(got == want)}/{want.size} identical", flush=True)
