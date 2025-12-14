# 🎨 Channel Flow Visualization

## Understanding YOLOv8 Width Scaling

### Before Fix (Broken) ❌

```
┌─────────────────────────────────────────────────────────────┐
│                    YAML Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv[128] → C2f[128] → CBAM[?]                            │
│       ↓         ↓          ↓                                │
│     WRONG!   Output     Expected                            │
│              channels   channels                            │
│              from C2f   by CBAM                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Actual Runtime (width_multiple = 0.25)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv → outputs 32 channels (128 * 0.25)                   │
│     ↓                                                       │
│  C2f → outputs 32 channels (128 * 0.25)                    │
│     ↓                                                       │
│  CBAM → expects 128 channels (NOT SCALED!)                 │
│     ↓                                                       │
│  ❌ ERROR: Expected 128 channels, got 32                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## After Fix (Working) ✅

```
┌─────────────────────────────────────────────────────────────┐
│                    YAML Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv[128] → C2f[128] → CBAM[128]                          │
│       ↓         ↓          ↓                                │
│     CORRECT! All use same channel spec                     │
│              All get scaled together                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Actual Runtime (width_multiple = 0.25)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv → outputs 32 channels (128 * 0.25)                   │
│     ↓                                                       │
│  C2f → outputs 32 channels (128 * 0.25)                    │
│     ↓                                                       │
│  CBAM → expects 32 channels (128 * 0.25)                   │
│     ↓                                                       │
│  ✅ SUCCESS: Channels match! (32 == 32)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Backbone Channel Flow

### YOLOv8n-CBAM (width_multiple = 0.25)

```
Layer#  Module      YAML Spec   →  Actual Channels  →  Tensor Shape
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0     Conv        [64]        →  16 (64*0.25)     →  (B, 16, 320, 320)
  1     Conv        [128]       →  32 (128*0.25)    →  (B, 32, 160, 160)

┌───────────────── P2/4 Feature Level ─────────────────┐
│  2     C2f         [128]       →  32               →  (B, 32, 160, 160) │
│  3     CBAM        [128]       →  32 ✅            →  (B, 32, 160, 160) │
└──────────────────────────────────────────────────────┘

  4     Conv        [256]       →  64 (256*0.25)    →  (B, 64, 80, 80)

┌───────────────── P3/8 Feature Level ─────────────────┐
│  5     C2f         [256]       →  64               →  (B, 64, 80, 80)   │
│  6     CBAM        [256]       →  64 ✅            →  (B, 64, 80, 80)   │
└──────────────────────────────────────────────────────┘

  7     Conv        [512]       →  128 (512*0.25)   →  (B, 128, 40, 40)

┌───────────────── P4/16 Feature Level ────────────────┐
│  8     C2f         [512]       →  128              →  (B, 128, 40, 40)  │
│  9     CBAM        [512]       →  128 ✅           →  (B, 128, 40, 40)  │
└──────────────────────────────────────────────────────┘

 10     Conv        [1024]      →  256 (max=1024)   →  (B, 256, 20, 20)

┌───────────────── P5/32 Feature Level ────────────────┐
│ 11     C2f         [1024]      →  256              →  (B, 256, 20, 20)  │
│ 12     CBAM        [1024]      →  256 ✅           →  (B, 256, 20, 20)  │
└──────────────────────────────────────────────────────┘

 13     SPPF        [1024]      →  256              →  (B, 256, 20, 20)

```

**Legend:**

- `✅` = Channel count matches between C2f output and CBAM input
- `YAML Spec` = What you write in the architecture file
- `Actual Channels` = What the model actually uses (after scaling)
- `(B, C, H, W)` = Batch, Channels, Height, Width

---

## CBAM Internal Structure

```
Input Tensor
(B, C, H, W)
     │
     ├──────────────────┬──────────────────┐
     │                  │                  │
     │         ┌────────▼────────┐         │
     │         │ Channel Attention│         │
     │         │                 │         │
     │         │ AvgPool → FC → ReLU      │
     │         │              ↓           │
     │         │ MaxPool → FC → ReLU      │
     │         │              ↓           │
     │         │      Add + Sigmoid       │
     │         └────────┬────────┘         │
     │                  │                  │
     │                  ▼                  │
     │         Element-wise Multiply       │
     │                  │                  │
     │         ┌────────▼────────┐         │
     │         │ Spatial Attention│         │
     │         │                 │         │
     │         │ AvgPool(dim=1)  │         │
     │         │       +         │         │
     │         │ MaxPool(dim=1)  │         │
     │         │       ↓         │         │
     │         │  Conv7x7 + Sigmoid        │
     │         └────────┬────────┘         │
     │                  │                  │
     │                  ▼                  │
     │         Element-wise Multiply       │
     │                  │                  │
     │                  ▼                  │
     └──────────────>  Add (residual) ─────┤
                       │                   │
                       ▼                   │
                  Output Tensor            │
                  (B, C, H, W) ◄───────────┘
                  Same shape as input!
```

**Key Points:**

1. **Input channels = Output channels** (CBAM doesn't change dimensions)
2. **Channel Attention**: Learns "what" features are important
3. **Spatial Attention**: Learns "where" features are important
4. **Residual connection**: Preserves original information

---

## Width Scaling Examples

### Different Model Variants

```
Model   Width     YAML [128]   Actual    YAML [256]   Actual
        Multiple                Channels               Channels
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLOv8n   0.25      [128]   →    32        [256]   →    64
YOLOv8s   0.50      [128]   →    64        [256]   →   128
YOLOv8m   0.75      [128]   →    96        [256]   →   192
YOLOv8l   1.00      [128]   →   128        [256]   →   256
YOLOv8x   1.25      [128]   →   160        [256]   →   320
```

**This is why using the same channel spec for C2f and CBAM works!**

- Both get scaled by the same factor
- Actual channel counts always match
- Architecture works for all model sizes (n/s/m/l/x)

---

## Common Pitfalls (Avoided in Our Implementation)

### ❌ Pitfall 1: Hardcoding Actual Channel Counts

```yaml
# WRONG - will break with different model sizes!
backbone:
  - [-1, 3, "C2f", [128, True]]
  - [-1, 1, "CBAM", [32]] # Hardcoded for 'n' variant only!
```

### ✅ Solution: Use Scaled Values

```yaml
# CORRECT - works for all model sizes
backbone:
  - [-1, 3, "C2f", [128, True]]
  - [-1, 1, "CBAM", [128]] # Same as C2f, both get scaled!
```

---

### ❌ Pitfall 2: Not Validating Channels in forward()

```python
# WRONG - silent errors, hard to debug
def forward(self, x):
    x = x * self.channel_attention(x)  # What if shapes don't match?
    return x
```

### ✅ Solution: Comprehensive Validation

```python
# CORRECT - catches errors early with clear messages
def forward(self, x):
    if x.size(1) != self.c1:
        raise ValueError(
            f"Channel mismatch! Expected {self.c1}, got {x.size(1)}. "
            f"Check YAML architecture."
        )
    x = x * self.channel_attention(x)
    return x
```

---

### ❌ Pitfall 3: Assuming CBAM Changes Channels

```python
# WRONG - CBAM is attention, not transformation!
class CBAM(nn.Module):
    def __init__(self, c1, c2):
        self.conv = nn.Conv2d(c1, c2, 1)  # NO! Don't change channels
```

### ✅ Solution: Preserve Channels

```python
# CORRECT - attention preserves shape
class CBAM(nn.Module):
    def __init__(self, c1, c2=None):
        if c2 is not None and c2 != c1:
            warnings.warn(f"CBAM preserves channels, c2 ignored")
        self.c2 = c1  # Output = Input channels
```

---

## Testing Channel Flow

### Quick Test Script

```python
# Test if channels match throughout the model
import torch
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-cbam.yaml')

# Test input
x = torch.randn(1, 3, 640, 640)

# Forward pass with hooks to print shapes
def print_shape(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            print(f"{name:20s}: {tuple(output.shape)}")
    return hook

# Register hooks
for name, module in model.model.named_modules():
    if 'CBAM' in str(type(module)):
        module.register_forward_hook(print_shape(name))

# Run
with torch.no_grad():
    output = model.model(x)

# Expected output:
# model.2.cbam      : (1, 32, 160, 160)   ✓
# model.5.cbam      : (1, 64, 80, 80)     ✓
# model.8.cbam      : (1, 128, 40, 40)    ✓
# model.11.cbam     : (1, 256, 20, 20)    ✓
```

---

## Summary

### The Fix in One Sentence

**Use the same channel specification in YAML for both C2f and CBAM, so they both get scaled by `width_multiple` and produce matching actual channel counts.**

### Why It Works

1. YOLOv8 applies `width_multiple` to **all** channel arguments
2. By using **matching specs**, both layers scale **together**
3. Actual runtime channels **always match**, regardless of model size

### Benefits

✅ Works for all YOLOv8 sizes (n/s/m/l/x)
✅ Self-documenting architecture
✅ Type-safe with validation
✅ Clear error messages
✅ Expert-level implementation

---

**Now you understand the root cause and solution!** 🎓

The notebook is fixed and ready to train. Happy training! 🚀
