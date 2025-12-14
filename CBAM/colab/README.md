# 🚀 Google Colab Training - YOLOv8 with CBAM

## Quick Start (5 Minutes)

### 1. Upload to Google Drive

1. Upload `YOLOv8_CBAM_Training.ipynb` to your Google Drive
2. Make sure your dataset is in: `/content/drive/MyDrive/YOLOv8 Traffic/dataset`

### 2. Open in Google Colab

1. Right-click the notebook → Open with → Google Colaboratory
2. **Important**: Change runtime to GPU
   - Runtime → Change runtime type → Hardware accelerator → **T4 GPU**

### 3. Run All Cells

- Click: Runtime → Run all
- Or press `Ctrl+F9`

**That's it!** The notebook will handle everything automatically.

---

## 📋 What This Notebook Does

### Automatic Setup

✅ Mounts Google Drive
✅ Installs Ultralytics YOLOv8
✅ Validates environment (Python, PyTorch, CUDA)
✅ Defines CBAM module with error handling
✅ Registers CBAM with Ultralytics
✅ Creates model architecture
✅ Validates dataset configuration
✅ Loads pretrained weights
✅ Tests forward pass

### Training

✅ Optimized for T4 GPU (16GB)
✅ Batch size: 16 (optimal for T4)
✅ Image size: 640×640
✅ Mixed precision training (faster)
✅ Automatic checkpointing
✅ Progress visualization
✅ Early stopping (patience: 50)

### Results Management

✅ Saves all results to Google Drive
✅ Copies best/last weights
✅ Saves training curves
✅ Saves confusion matrix
✅ Saves validation predictions
✅ Optional: Download best model

---

## ⚙️ Configuration

### Default Settings (Optimized for T4)

```python
Epochs: 100
Batch size: 16
Image size: 640×640
Learning rate: 0.01 → 0.0001
Mixed precision: Enabled
Early stopping: 50 epochs
```

### Customize Settings

Edit in **Step 9** of the notebook:

```python
TRAINING_CONFIG = {
    'epochs': 100,        # Change this
    'batch': 16,          # Reduce if OOM error
    'imgsz': 640,         # Or 416 for faster training
    # ... other settings
}
```

---

## 📂 Required Google Drive Structure

```
MyDrive/
└── YOLOv8 Traffic/
    ├── dataset/
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   └── labels/
    │       ├── train/
    │       └── val/
    └── data.yaml
```

### data.yaml Example

```yaml
path: /content/drive/MyDrive/YOLOv8 Traffic/dataset
train: images/train
val: images/val
nc: 10
names:
  0: bicycle
  1: bus
  2: car
  3: cng
  4: auto
  5: bike
  6: Multi-Class
  7: rickshaw
  8: truck
  9: van
```

---

## 🎯 Expected Results

### Training Time

| Hardware   | Batch Size | Time per Epoch | Total (100 epochs) |
| ---------- | ---------- | -------------- | ------------------ |
| **T4 GPU** | 16         | ~2-3 minutes   | **3-4 hours**      |
| T4 GPU     | 8          | ~4-5 minutes   | 6-8 hours          |
| CPU        | -          | Don't use!     | 20+ hours          |

### Performance Metrics

| Metric    | Expected Value |
| --------- | -------------- |
| mAP50     | 98-99%         |
| mAP50-95  | 89-91%         |
| Precision | 95-96%         |
| Recall    | 95-96%         |

---

## 📊 Output Files

After training, results are saved to:

```
Google Drive/YOLOv8 Traffic/training_results/yolov8n_cbam_bd_vehicles/
├── weights/
│   ├── best.pt          ← Use this for inference
│   └── last.pt          ← Use this to resume training
├── results.png          ← Training curves (loss, mAP, etc.)
├── results.csv          ← Numerical results per epoch
├── confusion_matrix.png ← Per-class performance
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
└── val_batch*.jpg       ← Validation predictions
```

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution** (Edit Step 9):

```python
TRAINING_CONFIG = {
    'batch': 8,        # Reduce from 16 to 8
    'imgsz': 416,      # Reduce from 640 to 416
    # ... rest of config
}
```

### Issue 2: Runtime Disconnected

**Cause**: Colab disconnects after ~90 minutes of inactivity

**Solutions**:

1. **Colab Pro**: $10/month for longer sessions
2. **Keep active**: Click around periodically
3. **Resume training**: Load `last.pt` and set `resume=True`

### Issue 3: Dataset Not Found

**Error**: `data.yaml not found`

**Solution**:

1. Check path in **Step 3**:

```python
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/YOLOv8 Traffic'  # Update this
```

2. Verify folder exists in Google Drive
3. Re-run from **Step 3**

### Issue 4: Import Errors

**Error**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:

- Re-run **Step 2** (Install Dependencies)
- Wait for installation to complete
- Check ✓ marks in output

### Issue 5: Slow Training

**Cause**: Using CPU instead of GPU

**Solution**:

1. Runtime → Change runtime type
2. Hardware accelerator → **T4 GPU**
3. Click Save
4. Runtime → Restart runtime
5. Re-run all cells

---

## 💡 Pro Tips

### 1. Monitor Training

Watch the progress bar and metrics:

- Loss should decrease steadily
- mAP should increase
- Check every 10-20 epochs

### 2. Save Intermediate Results

Training auto-saves checkpoints:

- `best.pt`: Best model so far (highest mAP)
- `last.pt`: Most recent checkpoint

### 3. Use Colab Resources Wisely

- T4 GPU is free but limited
- Typical limit: ~12 hours per day
- Don't waste time on CPU training

### 4. Resume If Disconnected

If training stops, resume with:

```python
from ultralytics import YOLO
model = YOLO('path/to/last.pt')
model.train(resume=True)
```

### 5. Test Before Full Training

Quick test (5 epochs):

```python
TRAINING_CONFIG['epochs'] = 5
```

Run full training if results look good.

---

## 📱 Access Results Anywhere

Results are saved to **Google Drive**, so you can:

1. **View on phone**: Open Google Drive app
2. **Share**: Share Drive folder with team
3. **Download**: Download specific files
4. **Backup**: Already backed up by Google

---

## 🎓 Understanding the Notebook

### Cell-by-Cell Breakdown

| Step | What It Does        | Time     | Can Skip?                   |
| ---- | ------------------- | -------- | --------------------------- |
| 1    | Mount Drive         | 5s       | ❌ No                       |
| 2    | Install packages    | 30s      | ❌ No                       |
| 3    | Configure paths     | 1s       | ❌ No                       |
| 4    | Define CBAM         | 2s       | ❌ No                       |
| 5    | Register CBAM       | 1s       | ❌ No                       |
| 6    | Create architecture | 1s       | ❌ No                       |
| 7    | Validate dataset    | 5s       | ⚠️ Optional but recommended |
| 8    | Load model          | 10s      | ❌ No                       |
| 9    | Configure training  | 1s       | ❌ No                       |
| 10   | **Train model**     | **3-4h** | ❌ No                       |
| 11   | Validate model      | 2m       | ⚠️ Optional but recommended |
| 12   | Copy to Drive       | 1m       | ⚠️ Recommended              |
| 13   | Download model      | -        | ✅ Optional                 |
| 14   | Test inference      | 1m       | ✅ Optional                 |

---

## 🔄 Resume Training

If your session disconnects:

1. Re-run cells 1-5 (setup)
2. Modify cell 10:

```python
# Load last checkpoint
model = YOLO(f'{RESULTS_DIR}/detect/{TRAINING_CONFIG["name"]}/weights/last.pt')

# Resume training
results = model.train(resume=True)
```

---

## 📞 Support

### Common Questions

**Q: How much does this cost?**
A: Free with Google Colab! (T4 GPU included)

**Q: Can I train overnight?**
A: Free Colab may disconnect. Consider Colab Pro for longer sessions.

**Q: Where are results saved?**
A: Google Drive → YOLOv8 Traffic → training_results

**Q: Can I use this notebook multiple times?**
A: Yes! It will overwrite previous results (set `exist_ok=True`)

**Q: What if I get an error?**
A: Check the Troubleshooting section above

---

## ✅ Pre-Flight Checklist

Before running the notebook:

- [ ] T4 GPU runtime selected
- [ ] Google Drive mounted successfully
- [ ] Dataset in correct location
- [ ] `data.yaml` exists and is correct
- [ ] At least 3-4 hours available
- [ ] Stable internet connection

---

## 🎉 After Training

### 1. Review Results

Open these files in Google Drive:

- `results.png` - Training curves
- `confusion_matrix.png` - Per-class accuracy

### 2. Test Your Model

```python
from ultralytics import YOLO

# Load best model
model = YOLO('path/to/best.pt')

# Run on image
results = model('path/to/image.jpg')

# Show results
results[0].show()
```

### 3. Deploy

Export for production:

```python
model.export(format='onnx')       # Cross-platform
model.export(format='tflite')     # Mobile/edge devices
model.export(format='engine')     # NVIDIA TensorRT
```

---

## 🌟 Why This Notebook Is Robust

### Error Prevention

✅ **Input validation**: Checks all inputs before processing
✅ **File existence**: Verifies files exist before loading
✅ **GPU detection**: Auto-detects and uses GPU
✅ **Memory management**: Optimized for T4 GPU
✅ **Error messages**: Clear, actionable error messages

### Best Practices

✅ **Expert CBAM implementation**: Follows original paper
✅ **Proper initialization**: Kaiming & Xavier initialization
✅ **Mixed precision**: Faster training on T4
✅ **Auto checkpointing**: Never lose progress
✅ **Result backup**: Auto-saves to Drive

### User-Friendly

✅ **Single click**: Run all cells, wait for results
✅ **Progress bars**: See training progress
✅ **No coding needed**: Just update paths
✅ **Comments**: Every step explained
✅ **Error handling**: Graceful failure with solutions

---

## 📚 Additional Resources

- **CBAM Paper**: https://arxiv.org/abs/1807.06521
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **Colab FAQ**: https://research.google.com/colaboratory/faq.html

---

**Ready to train? Upload the notebook to Colab and click Run All! 🚀**

**Expected outcome**: A state-of-the-art vehicle detector trained in 3-4 hours!
