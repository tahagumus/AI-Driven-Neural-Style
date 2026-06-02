# AI-Driven Neural Style Transfer

## Project Overview

This project was developed as part of an internship program. The goal is to apply neural style transfer techniques to video content while preserving temporal consistency between consecutive frames.

## Objective

Traditional neural style transfer methods often produce flickering effects when applied frame-by-frame to videos. This project aims to reduce these visual inconsistencies by incorporating temporal consistency mechanisms, resulting in smoother and more visually coherent stylized videos.

## Features

* AI-based neural style transfer
* Video frame processing
* Temporal consistency preservation
* Custom style image support
* Automated video stylization pipeline

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy

## Project Structure

* `mainn.py` – Main execution script
* `style_transfer.py` – Neural style transfer implementation
* `temporal_consistency.py` – Temporal consistency processing
* `requirements.txt` – Required dependencies
* `starry night.jpg` – Example style image
* `firstvideo.mp4` – Sample input video
* `output/` – Directory containing the generated stylized video

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

Navigate to the project directory and run:

```bash
python mainn.py --style "starry night.jpg" --video firstvideo.mp4
```

## Input Files

* Style image: `starry night.jpg`
* Input video: `firstvideo.mp4`

## Output

After processing is completed, the generated stylized video will be saved in the `output/` directory.

Example:

```text
output/
```

The final processed video can be found inside the `output` folder.

## Results

The system applies the selected artistic style to the input video while maintaining consistency between frames to reduce flickering and improve visual quality.

## Internship Information

This project was developed during a software engineering internship and focuses on applying artificial intelligence techniques for video style transfer while maintaining visual consistency between frames.

## Authors

* Taha GÜMÜŞ
* Emir BALTACI
