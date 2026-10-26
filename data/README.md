# data

## maya-expert-profile.bin

Routed-expert pick counts recorded by Project Maya while serving DS4.1-1017 (EXL3 3.5 bpw)
to the owner's agent harness, copied from the Maya pack on 2026-10-26 before it was deleted.

- Layout: little-endian `uint32[43][384]` — 43 router layers (40 model layers, then 3 DSpark
  draft blocks; verify the mapping against Maya's `save_profile` before relying on rows 40–42)
  × 384 routed Experts. 66,048 bytes.
- sha256 `ab679869a83bc58407997e99fd0c96005627e7d9713a1c275dc296228f30b7e1`
- Routing came from the EXL3-quantized model, so counts approximate native routing.
- Use: seed the Expert cache popularity prior and the boot-time preload set; estimate hit
  rate versus Slot count before any engine exists.
