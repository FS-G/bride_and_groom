# Bride and Groom Identification System

**DISCLAIMER - AMINDAV PROPERTY**

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.

## System Overview

The Bride and Groom Identification System is an advanced AI-powered solution designed to automatically process wedding videos and extract high-quality images of brides, grooms, and family moments. The system integrates with Trello for project management and uses multiple AI models for video classification, face detection, and image processing.

## Architecture

The system consists of three main components that can run in parallel:

### 1. Main Application (`app.py`)
- **Purpose**: Core video processing and bride/groom identification
- **Functionality**: 
  - Monitors Trello "IN" list for new projects
  - Categorizes videos (ceremony, dance/party, other)
  - Detects tripod-stabilized videos
  - Extracts bride and groom faces from ceremony videos
  - Generates family photos from bride-groom interactions
  - Uploads results to Trello cards

### 2. Face Processing API (`api.py`)
- **Purpose**: Parallel face verification and grouping service
- **Functionality**:
  - Receives face images from main application
  - Uses DeepFace for face verification and grouping
  - Implements adaptive thresholding for optimal grouping
  - Returns best quality face groups

### 3. Model Fine-tuning (`finetune.py`)
- **Purpose**: AI model improvement and training
- **Functionality**:
  - Extracts frames from specified videos
  - Fine-tunes CNN models on new data
  - Updates model versions automatically
  - Updates system parameters

## System Components

### Core Modules (`utils/`)

#### `trello_manager.py`
- Trello API integration for project management
- Card movement between lists (IN → PROCESS → OUT/ERROR)
- File attachment uploads
- Status message writing

#### `bride_groom_extractor.py`
- YOLO-based person detection (bride/groom)
- Face extraction from detected persons
- Bounding box optimization
- Confidence scoring

#### `video_finder.py`
- Searches multiple directories for project videos
- Filters videos by length and criteria
- Organizes video paths by project

#### `video_label.py`
- CNN-based video classification
- Categorizes videos as ceremony, dance/party, or other
- Frame-based analysis for accurate classification

#### `tripod_detector.py`
- Motion analysis for tripod detection
- Identifies stable video segments
- Helps prioritize high-quality footage

#### `family_predictor.py`
- AI model for predicting best family photo moments
- Analyzes bride-groom interaction frames
- Selects optimal images for family photos

#### `image_saver.py`
- Local image storage management
- Organizes images by project and category
- Handles file naming and directory structure

#### `textfile_writer.py`
- Documentation generation
- Creates text files with video categories
- Records tripod video information

## Configuration Files

### `parameters.json`
Main system configuration containing:
- Trello board and list names
- Base paths for video search
- Model IDs for different AI models
- Processing thresholds and parameters
- File length requirements

### `finetune_parameters.json`
Fine-tuning configuration containing:
- Video targets for training data extraction
- Target categories for each video
- Training data organization

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Trello API credentials
- Access to video storage directories

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Model Files
Ensure the following model files are present:
- `models/cnn_model_weights{version}.weights.h5` - Video classification model
- `models/cnn_model_family_weights1.weights.h5` - Family photo prediction model
- `models_yolo/base{version}.pt` - Bride/groom detection model
- `models_yolo/face{version}.pt` - Face detection model

## Usage

### Starting the System

#### Option 1: Using Batch Files (Windows)
```bash
# Start main application and API in parallel
start.bat

# Start fine-tuning process
stare_finetune.bat
```

#### Option 2: Manual Startup
```bash
# Terminal 1: Start main application
python app.py

# Terminal 2: Start face processing API
python api.py

# Terminal 3: Run fine-tuning (when needed)
python finetune.py
```

### Workflow

1. **Project Initiation**
   - Add project card to Trello "IN" list
   - Card name should match video directory structure
   - System automatically detects new cards

2. **Video Processing**
   - System finds videos matching project criteria
   - Categorizes videos by type (ceremony, dance, other)
   - Identifies tripod-stabilized videos

3. **Face Extraction**
   - Processes ceremony videos for bride/groom detection
   - Extracts face images using YOLO models
   - Sends faces to API for verification and grouping

4. **Result Generation**
   - Creates family photos from bride-groom interactions
   - Saves images locally and uploads to Trello
   - Moves card to "OUT" list upon completion

5. **Error Handling**
   - Failed projects moved to "ERROR" list
   - Error messages written to cards
   - System continues with next project

### Fine-tuning Process

When model performance needs improvement:

1. **Update Configuration**
   - Edit `finetune_parameters.json`
   - Add video paths and target categories
   - Specify training data organization

2. **Run Fine-tuning**
   ```bash
   python finetune.py
   ```

3. **Automatic Updates**
   - System extracts frames from specified videos
   - Fine-tunes CNN model on new data
   - Updates `parameters.json` with new model ID
   - Preserves previous model versions

## Directory Structure

```
bride_and_groom_finalized/
├── app.py                          # Main application entry point
├── api.py                          # Face processing API
├── finetune.py                     # Model fine-tuning script
├── parameters.json                 # Main system configuration
├── finetune_parameters.json        # Fine-tuning configuration
├── requirements.txt                # Python dependencies
├── start.bat                       # Windows startup script
├── stare_finetune.bat              # Windows fine-tuning script
├── models/                         # CNN model weights
├── models_yolo/                    # YOLO model weights
├── finetune/                       # Fine-tuning data and results
│   └── data/                       # Extracted training frames
├── temp_images/                    # Temporary API processing files
└── utils/                          # Core system modules
    ├── trello_manager.py           # Trello integration
    ├── bride_groom_extractor.py    # Face extraction
    ├── video_finder.py             # Video discovery
    ├── video_label.py              # Video classification
    ├── tripod_detector.py          # Tripod detection
    ├── family_predictor.py         # Family photo prediction
    ├── image_saver.py              # Image storage
    └── textfile_writer.py          # Documentation
```

## Configuration

### Trello Setup
1. Create Trello board with lists:
   - "Identify Bride Groom IN"
   - "Identify Bride Groom PROCESS"
   - "Identify Bride Groom OUT"
   - "Identify Bride Groom ERROR"

2. Update API credentials in `app.py`:
   ```python
   API_KEY = "your_api_key"
   API_SECRET = "your_api_secret"
   TOKEN = "your_token"
   TOKEN_SECRET = "your_token_secret"
   ```

### Video Directory Structure
Videos should be organized in directories matching Trello card names:
```
base_paths/
└── project_name/
    └── video_files.mp4
```

### Model Configuration
Update model IDs in `parameters.json`:
```json
{
  "model_id_video": 20,
  "model_id_bg": 6,
  "model_id_face": 11
}
```

## Troubleshooting

### Common Issues

1. **No videos found**
   - Check `base_paths` in `parameters.json`
   - Verify video directory structure matches card names
   - Ensure videos meet length requirements

2. **Face extraction failures**
   - Check YOLO model files exist
   - Verify model IDs in configuration
   - Adjust confidence thresholds if needed

3. **Trello API errors**
   - Verify API credentials
   - Check board and list names
   - Ensure proper permissions

4. **Fine-tuning issues**
   - Verify video paths in `finetune_parameters.json`
   - Check available disk space for frame extraction
   - Ensure GPU memory for training

### Logging
- Main application logs to console
- API service provides request/response logging
- Fine-tuning script shows detailed progress

## Performance Considerations

- **GPU Usage**: YOLO models benefit from CUDA acceleration
- **Memory**: Large videos may require significant RAM
- **Storage**: Frame extraction and image storage need adequate disk space
- **Network**: Trello API calls require stable internet connection

## Security Notes

- API credentials are hardcoded and should be secured
- Video paths may contain sensitive information
- Temporary files are created during processing
- Consider data retention policies

## Support

For technical support or questions about the system, contact the Amindav Development Team.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Amindav Development Team 