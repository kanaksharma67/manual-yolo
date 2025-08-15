# Poker YOLO Detector with GPT-4o Integration

This project combines YOLO object detection with GPT-4o vision analysis to accurately detect and extract poker table information from screenshots or live capture.

## 🚀 Features

- **YOLO Object Detection**: Detects poker table elements (cards, buttons, player info)
- **Local OCR**: Uses EasyOCR for text extraction
- **GPT-4o Fallback**: When local OCR fails, sends cropped images to GPT-4o for better accuracy
- **Live Capture**: Real-time screen monitoring with automatic updates
- **Clean JSON Output**: Structured data in a simple, readable format
- **Collage Generation**: Creates image collages of detected regions for GPT-4o analysis

## 📋 Requirements

- Python 3.10+
- OpenAI API key (for GPT-4o integration)
- CUDA-compatible GPU (optional, for faster processing)

## 🛠️ Installation

### 1. Clone and Setup
```bash
cd manual-yolo
```

### 2. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API Key
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "your_api_key_here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY="your_api_key_here"

# Permanent (Windows)
setx OPENAI_API_KEY "your_api_key_here"

# Permanent (Linux/Mac)
echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.bashrc
```

## 🎯 Quick Start

### Option 1: Use Setup Scripts
```bash
# Windows
setup_env.bat

# PowerShell
.\setup_env.ps1
```

### Option 2: Manual Setup
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Test the system
python test_yolo.py

# Run the detector
python yolo.py
```

## 📊 Output Format

The system generates a clean, structured JSON output:

```json
{
    "game_info": {
        "game_id": "12345",
        "time": "2025-01-27 15:30:45",
        "game_state": "FLOP"
    },
    "my_cards": {
        "card1": "As",
        "card2": "Kh"
    },
    "my_info": {
        "stack": "1500",
        "bet": "100"
    },
    "villains": [
        {
            "name": "Player1",
            "stack": "2000",
            "bet": "150"
        },
        {
            "name": "Player2",
            "stack": "1200",
            "bet": "0"
        }
    ],
    "community_cards": ["7c", "9d", "2h"],
    "buttons": ["Fold", "Call", "Raise"],
    "pot": "450"
}
```

## 🔧 Configuration

Edit the configuration section in `yolo.py`:

```python
# Model paths
MODEL_PATH = "poker_model.pt"
DATASET_YAML = "roadmap1.v3i.yolov8/data.yaml"

# Capture settings
FULLSCREEN = True
SCREEN_REGION = (100, 100, 1000, 700)
CONFIDENCE_THRESHOLD = 0.5

# OCR and GPT settings
USE_OCR = True
USE_GPT_FALLBACK = True
GPT_IMAGE_MODEL = "gpt-4o"
```

## 🎮 How It Works

1. **Detection**: YOLO model detects poker table elements
2. **Local OCR**: EasyOCR attempts to extract text from detected regions
3. **Collage Creation**: If important fields are missing, creates a collage of cropped regions
4. **GPT-4o Analysis**: Sends collage to GPT-4o for better text recognition
5. **Data Fusion**: Combines local OCR and GPT-4o results
6. **JSON Output**: Generates clean, structured output

## 📁 File Structure

```
manual-yolo/
├── yolo.py                 # Main detector script
├── test_yolo.py           # Test script
├── setup_env.bat          # Windows setup script
├── setup_env.ps1          # PowerShell setup script
├── requirements.txt        # Python dependencies
├── poker_model.pt         # YOLO model file
├── custom_ocr_models/     # Custom OCR models
└── roadmap1.v3i.yolov8/   # Dataset and training files
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model not loaded" error**
   - Check if `poker_model.pt` exists
   - Verify model file is not corrupted

2. **OpenAI API errors**
   - Verify API key is set correctly
   - Check API key has sufficient credits
   - Ensure GPT-4o access is enabled

3. **OCR failures**
   - Install custom OCR models in `custom_ocr_models/`
   - Check image quality and resolution

4. **Import errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

### Debug Mode

Enable debug mode in `yolo.py`:
```python
DEBUG = True
```

This will save cropped images and collages for inspection.

## 🔄 Live Mode

The system runs in live mode by default, continuously monitoring your screen:

- **Fullscreen**: Captures entire screen
- **Region**: Captures specific screen region
- **Auto-save**: Updates JSON and annotated image every 2 seconds
- **Stop**: Press `Ctrl+C` to stop

## 📈 Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for faster YOLO inference
2. **Model Optimization**: Use smaller YOLO models for faster processing
3. **Screen Resolution**: Lower resolution screens process faster
4. **Update Frequency**: Adjust `LOOP_INTERVAL_SECONDS` for performance vs. accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review error messages in the terminal
3. Enable debug mode for more information
4. Check that all dependencies are properly installed

## 🎯 Future Enhancements

- [ ] Support for multiple poker sites
- [ ] Hand history analysis
- [ ] Real-time statistics
- [ ] Multi-language support
- [ ] Custom model training interface
