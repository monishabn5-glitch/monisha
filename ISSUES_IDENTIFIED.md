# Critical Issues Found in Deepfake Detection Code

## 🔴 MAJOR PROBLEMS

### 1. **Missing PreTrained Weights in ResNet18** (CRITICAL)
**Location:** `model.py` line 13
```python
backbone = models.resnet18(weights=None)  # ❌ WRONG!
```
**Problem:** ResNet18 is initialized with random weights instead of ImageNet pretrained weights. This severely cripples feature extraction.
**Impact:** Features are meaningless → Model can't detect deepfakes properly

**Fix:** Use `weights=models.ResNet18_Weights.DEFAULT`

---

### 2. **Inconsistent Normalization Between Training & Inference** 
**Problem:** 
- Training likely used different augmentation/normalization
- Current code may use wrong mean/std values
- No data augmentation during training code visible

**Fix:** Ensure consistent preprocessing with training pipeline

---

### 3. **LSTM Output Selection Bug**
**Location:** `model.py` line 52
```python
return out[:, -1, :]  # ❌ Only takes LAST frame features
```
**Problem:** LSTM output from last timestep may contain temporal artifacts from padding
**Better Approach:** Use hidden state or pool LSTM outputs properly

---

### 4. **Attention Fusion Logic Issues**
**Location:** `model.py` line 38-44
```python
stacked = torch.stack([f, e, m, c], dim=1)  # Shape: (B, 4, feat_dim)
weights = self.attn_weights(concat).unsqueeze(-1)  # Shape: (B, 4, 1)
return (stacked * weights).sum(dim=1)  # May cause broadcasting issues
```
**Problem:** Attention weights might not properly normalize or emphasize important features

---

### 5. **No Temporal Consistency Check**
**Problem:** 
- Takes only 30 sampled frames from entire video
- No sliding window or overlapping frame analysis
- Single prediction per video with no confidence averaging

**Fix:** Process multiple temporal windows and average predictions

---

### 6. **Eye Region Extraction Bug**
**Location:** `inference.py` line 42-44
```python
ex2 = max(left_eye[0], right_eye[0]) + 25
ey2 = max(left_eye[1], right_eye[1]) + 25
eye = rgb[ey1:ey2, ex1:ex2]  # ❌ Coordinates may be invalid
```
**Problem:** Keypoint coordinates are in original frame space but being applied to face region
**Result:** Eye region might be completely misaligned

---

### 7. **Mouth Region Extraction Issues**
**Location:** `inference.py` line 46-51
```python
mouth_left, mouth_right = kp['mouth_left'], kp['mouth_right']
mx1 = max(0, ((mouth_left[0] + mouth_right[0]) // 2) - 45)
my1 = max(0, ((mouth_left[1] + mouth_right[1]) // 2) - 30)
```
**Problem:** Same coordinate system mismatch. Mouth region coordinates are in full frame space, not relative to face crop

---

### 8. **No Face Quality Checks**
**Problem:** 
- Accepts any detected face regardless of quality
- No minimum face size threshold
- No occlusion detection
- Could be processing faces that are too small or poorly detected

---

### 9. **Padding Strategy for Short Videos**
**Location:** `inference.py` line 73-76
```python
while len(faces_t) < SEQ_LEN:
    faces_t.append(faces_t[-1])  # ❌ Duplicates last frame
    eyes_t.append(eyes_t[-1])
    mouths_t.append(mouths_t[-1])
```
**Problem:** 
- Duplicate padding creates artificial temporal patterns
- LSTM may learn to exploit this as a fake indicator
- Videos with fewer frames get penalized

---

### 10. **No Inference Mode Setting**
**Problem:** Model uses BatchNorm during inference, which uses running statistics from training
- Could cause distribution shift
- BatchNorm dropout might introduce randomness

---

### 11. **Overconfident Softmax**
**Problem:** Using raw softmax without temperature scaling may give overconfident predictions

---

## 🎯 SUMMARY OF FIXES NEEDED

| Issue | Severity | Fix |
|-------|----------|-----|
| PreTrained weights | 🔴 CRITICAL | Add `weights=models.ResNet18_Weights.DEFAULT` |
| Region coordinate mismatch | 🔴 CRITICAL | Convert keypoints to face-relative coordinates |
| LSTM output selection | 🟠 HIGH | Use hidden state properly |
| Padding strategy | 🟠 HIGH | Use interpolation instead of duplication |
| Temporal consistency | 🟠 HIGH | Add sliding window predictions |
| Attention fusion | 🟡 MEDIUM | Improve normalization & weighting |
| Face quality checks | 🟡 MEDIUM | Add validation filters |
| BatchNorm eval mode | 🟡 MEDIUM | Ensure model.eval() is set |
| Missing augmentation details | 🟡 MEDIUM | Document training preprocessing |

---

## Why 90% Training Accuracy But Poor Real-World Performance?

1. **Overfitting to training data distribution** → Test data differs
2. **Incorrect preprocessing mismatch** → Features don't align
3. **Region extraction bugs** → Eyes/mouth may be from wrong locations
4. **LSTM exploiting padding patterns** → Padding creates artificial deepfake signatures
5. **BatchNorm statistics mismatch** → Inference uses different distribution
