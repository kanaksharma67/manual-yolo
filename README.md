# Poker YOLO Detector with Rank Classification

This project combines YOLO object detection with custom rank classification to accurately detect and extract poker table information from screenshots or live capture.

## 🚀 Features

- **YOLO Object Detection**: Detects poker table elements (cards, buttons, player info)
- **Custom Rank Classifier**: Trained model for accurate card rank recognition
- **Local OCR**: Uses EasyOCR for text extraction
- **Live Capture**: Real-time screen monitoring with automatic updates
- **Clean JSON Output**: Structured data in a simple, readable format
- **Screen Coordinate Detection**: Tool to find exact screen coordinates for your setup

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, for faster processing)
- Windows OS (tested on Windows 10/11)

## 🛠️ Installation & Setup

### 1. Clone and Navigate to Project
```bash
cd manual-yolo
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# OR for Command Prompt
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start Guide

### Step 1: Train the Rank Classifier
First, you need to train the rank classifier model:

1. **Update Project Path**: Open `class.py` and replace the project path:
   ```python
   project_base = r"C:\Users\HP\monday\manual-yolo"  # Replace with your exact path
   ```

2. **Run Training**:
   ```bash
   python class.py
   ```
   This will:
   - Train the model for 50 epochs
   - Save the trained model as `rank_classifier.pt`
   - Use the dataset in `rank_classifier/` folder

### Step 2: Get Screen Coordinates
Use the coordinate detection tool to find your screen coordinates:

```bash
python getcors.py
```

1. Move your mouse to different corners of your poker table
2. Note the X, Y coordinates displayed
3. Press `Ctrl+C` to stop when done

### Step 3: Configure Detection
Open `detect.py` and update the screen region with your coordinates:

```python
# Replace with your actual screen coordinates
SCREEN_REGION = {"top": 0, "left": 0, "width": 1919, "height": 1199}
```

### Step 4: Run Detection
```bash
python detect.py
```

This will:
- Start live screen capture
- Detect poker elements in real-time
- Save screenshots to `live_output/` folder
- Generate `detections.json` with game data
- Create `game1.json` for individual game tracking

## 📊 Output Structure

### Live Output Folder
The `live_output/` folder contains:
- **Screenshots**: Timestamped images of detected regions
- **detections.json**: Real-time detection data
- **game1.json**: Current game information

### Detection Data Format
```json
{
    "timestamp": "2025-01-27 15:30:45",
    "game_state": "FLOP",
    "hero_cards": {
        "card1": {"rank": "A", "suit": "spades"},
        "card2": {"rank": "K", "suit": "hearts"}
    },
    "community_cards": [
        {"rank": "7", "suit": "clubs"},
        {"rank": "9", "suit": "diamonds"},
        {"rank": "2", "suit": "hearts"}
    ],
    "pot": "450",
    "buttons": ["Fold", "Call", "Raise"]
}
```

## 🔧 Configuration

### Model Paths
- **Poker Detection**: `poker_model.pt` (YOLO model for table elements)
- **Rank Classification**: `rank_classifier.pt` (Custom trained rank classifier)

### Screen Capture Settings
```python
# Full screen capture
SCREEN_REGION = {"top": 0, "left": 0, "width": 1919, "height": 1199}

# Or specific region (example)
SCREEN_REGION = {"top": 100, "left": 100, "width": 800, "height": 600}
```

### Training Parameters (class.py)
```python
model.train(
    data=dataset_path,
    epochs=50,        # Training epochs
    imgsz=64,         # Image size
    batch=64,         # Batch size
    workers=4,        # Number of workers
    patience=10       # Early stopping patience
)
```

## 📁 Project Structure

```
manual-yolo/
├── class.py                 # Rank classifier training script
├── detect.py               # Main detection script
├── getcors.py              # Screen coordinate detection tool
├── requirements.txt        # Python dependencies
├── rank_classifier.pt      # Trained rank classifier model
├── poker_model.pt          # YOLO poker detection model
├── rank_classifier/        # Training dataset
│   ├── train/             # Training images by rank
│   └── valid/             # Validation images
├── live_output/            # Detection outputs
│   ├── screenshots/        # Captured images
│   ├── detections.json     # Real-time data
│   └── game1.json         # Current game data
└── runs/                   # Training outputs
    └── rank_classifier/    # Training results and weights
```

## 🎮 How It Works

1. **Training Phase**: 
   - `class.py` trains a YOLOv8 classification model on card rank images
   - Uses the `rank_classifier/` dataset with images organized by rank

2. **Detection Phase**:
   - `detect.py` captures screen in real-time
   - YOLO model detects poker table elements
   - Rank classifier identifies card ranks
   - OCR extracts text from buttons and other elements

3. **Output Generation**:
   - Saves annotated screenshots
   - Updates JSON files with detection results
   - Tracks game state changes

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure `rank_classifier.pt` exists (run `class.py` first)
   - Check `poker_model.pt` is present

2. **Import errors**
   - Activate virtual environment: `.\venv\Scripts\Activate.ps1`
   - Install dependencies: `pip install -r requirements.txt`

3. **Screen capture issues**
   - Use `getcors.py` to find correct coordinates
   - Update `SCREEN_REGION` in `detect.py`

4. **Training failures**
   - Check dataset structure in `rank_classifier/` folder
   - Verify Python 3.10+ is being used

### Debug Mode
Enable debug output in `detect.py`:
```python
DEBUG = True
```

## 📈 Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for faster processing
2. **Screen Region**: Limit capture area to poker table region only
3. **Update Frequency**: Adjust screenshot interval in `detect.py`
4. **Model Size**: Use smaller YOLO models for faster inference

## 🔄 Live Mode Features

- **Real-time Detection**: Continuous screen monitoring
- **Auto-save**: Automatic screenshot and JSON updates
- **Game Tracking**: Maintains game state across hands
- **Stop Detection**: Press `Ctrl+C` to stop

## 📈 Usage Workflow

1. **Setup**: Install dependencies and activate virtual environment
2. **Train**: Run `class.py` to train rank classifier
3. **Configure**: Use `getcors.py` to find screen coordinates
4. **Detect**: Run `detect.py` for live poker detection
5. **Monitor**: Check `live_output/` folder for results
6. **Analyze**: Review `detections.json` and `game1.json` for game data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure virtual environment is activated
4. Check file paths and coordinates are correct
5. Review error messages in the terminal

## 🎯 Future Enhancements

- [ ] Support for multiple poker sites
- [ ] Hand history analysis
- [ ] Real-time statistics dashboard
- [ ] Multi-language support
- [ ] Custom model training interface
- [ ] Integration with poker analysis tools
