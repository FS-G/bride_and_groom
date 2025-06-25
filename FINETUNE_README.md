# Model Fine-tuning Guide

**DISCLAIMER - AMINDAV PROPERTY**

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.

## Overview

The fine-tuning process (`finetune.py`) is used to improve the AI model's performance when it encounters issues with specific projects. This guide explains how to update the configuration, run the fine-tuning process, and restart the system.

## When to Use Fine-tuning

Fine-tuning should be performed when:
- The main application fails to properly categorize videos
- Face extraction quality is poor for specific projects
- Model performance degrades over time
- New video formats or styles are introduced

## Configuration Setup

### 1. Update finetune_parameters.json

**CRITICAL**: You must update the paths in `finetune_parameters.json` for the project folder that caused trouble.

```json
{
    "video_targets": [
      {
        "video_path": "PATH_TO_PROBLEMATIC_VIDEO_1.mp4",
        "target_folder": "ceremony"
      },
      {
        "video_path": "PATH_TO_PROBLEMATIC_VIDEO_2.mp4",
        "target_folder": "dance"
      },
      {
        "video_path": "PATH_TO_PROBLEMATIC_VIDEO_3.mp4",
        "target_folder": "other"
      }
    ]
}
```

### 2. Target Folder Categories

The system uses three target folders based on model identification:

#### `ceremony`
- **Purpose**: Videos of wedding ceremonies
- **Content**: Bride and groom at altar, vows, ring exchange, etc.
- **Use Case**: Primary source for bride/groom face extraction

#### `dance`
- **Purpose**: Videos of dancing and party scenes
- **Content**: First dance, party dancing, celebrations
- **Use Case**: Secondary content for family photo generation

#### `other`
- **Purpose**: Videos that don't fit ceremony or dance categories
- **Content**: Preparations, receptions, miscellaneous footage
- **Use Case**: Background content for comprehensive analysis

## Fine-tuning Process

### Step 1: Prepare Training Data

1. **Identify Problematic Videos**
   - Find videos from projects that caused issues
   - Note their correct categories (ceremony, dance, other)

2. **Update Configuration**
   - Open `finetune_parameters.json`
   - Replace video paths with problematic video paths
   - Set correct `target_folder` for each video

3. **Verify Paths**
   - Ensure all video paths are accessible
   - Check that videos are in supported formats (.mp4, .avi, .mkv, etc.)

### Step 2: Run Fine-tuning

```bash
# Option 1: Using batch file (Windows)
stare_finetune.bat

# Option 2: Direct execution
python finetune.py
```

### Step 3: Monitor Progress

The fine-tuning process will:

1. **Extract Frames**
   - Extract 50 frames from each video
   - Resize frames to 224x224 pixels
   - Save frames in categorized folders

2. **Train Model**
   - Load latest model weights
   - Fine-tune on new data
   - Freeze early layers for stability
   - Train for 5 epochs with low learning rate

3. **Update System**
   - Save new model weights with incremented version
   - Update `parameters.json` with new model ID
   - Preserve previous model versions

### Step 4: Restart Main Application

**ESSENTIAL**: After fine-tuning is complete, you must restart the main application.

1. **Stop Current Application**
   ```bash
   # Close the main application (Ctrl+C in terminal)
   # Or close the terminal running app.py
   ```

2. **Restart Using Batch File**
   ```bash
   # Click or run start.bat
   start.bat
   ```

3. **Verify Restart**
   - Check that app.py is running
   - Verify API service is active
   - Monitor for any startup errors

## Directory Structure for Fine-tuning

```
finetune/
├── data/
│   ├── ceremony/          # Extracted ceremony video frames
│   ├── dance/             # Extracted dance video frames
│   └── other/             # Extracted other video frames
└── models/
    └── cnn_model_weights{version}.weights.h5
```

## Configuration Examples

### Example 1: Ceremony Video Issues
```json
{
    "video_targets": [
      {
        "video_path": "P:/ppmaker/out/_+PROJECT+_Name/ceremony_video.mp4",
        "target_folder": "ceremony"
      }
    ]
}
```

### Example 2: Multiple Category Issues
```json
{
    "video_targets": [
      {
        "video_path": "H:/weddings/project1/ceremony.mp4",
        "target_folder": "ceremony"
      },
      {
        "video_path": "H:/weddings/project1/dance.mp4",
        "target_folder": "dance"
      },
      {
        "video_path": "H:/weddings/project1/prep.mp4",
        "target_folder": "other"
      }
    ]
}
```

## Troubleshooting

### Common Issues

1. **Video Path Not Found**
   - Verify video paths in `finetune_parameters.json`
   - Check file permissions
   - Ensure videos exist and are accessible

2. **Insufficient Training Data**
   - Add more videos to each category
   - Ensure balanced representation across categories
   - Minimum 10-20 videos per category recommended

3. **Model Training Errors**
   - Check available GPU memory
   - Verify TensorFlow installation
   - Ensure sufficient disk space for frame extraction

4. **Application Restart Issues**
   - Close all Python processes
   - Check for port conflicts (API service)
   - Verify model files are accessible

### Error Messages

- **"No matching directory found"**: Check video paths in configuration
- **"Data directory is empty"**: Verify video files exist and are accessible
- **"Model weights not found"**: Check models directory structure
- **"Parameters file not found"**: Verify parameters.json exists

## Best Practices

### Video Selection
- Choose representative videos from problematic projects
- Include diverse lighting conditions and angles
- Balance categories (ceremony, dance, other)
- Use high-quality videos when possible

### Configuration Management
- Backup `finetune_parameters.json` before changes
- Document which projects caused issues
- Keep track of model versions and performance
- Test with small datasets first

### System Maintenance
- Monitor model performance after fine-tuning
- Keep previous model versions for rollback
- Regular fine-tuning for new video styles
- Document successful configurations

## Performance Monitoring

After fine-tuning and restart:

1. **Test with New Projects**
   - Add test cards to Trello "IN" list
   - Monitor categorization accuracy
   - Check face extraction quality

2. **Compare Results**
   - Compare with previous model performance
   - Note improvements in problem areas
   - Document any new issues

3. **Iterative Improvement**
   - If issues persist, repeat fine-tuning
   - Add more problematic videos to training data
   - Adjust model parameters if needed

## File Locations

- **Configuration**: `finetune_parameters.json`
- **Training Data**: `finetune/data/`
- **Model Weights**: `models/cnn_model_weights{version}.weights.h5`
- **Main Parameters**: `parameters.json` (auto-updated)
- **Startup Script**: `start.bat`
- **Fine-tuning Script**: `stare_finetune.bat`

## Support

For issues with the fine-tuning process:
1. Check this guide for common solutions
2. Verify all file paths and permissions
3. Review error logs and console output
4. Contact Amindav Development Team

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Amindav Development Team 