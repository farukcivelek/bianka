# bianka
Vision-Based Identification of Plastic Parts via Fluorescent Particle Patterns
# Feature-Based Image Registration and Identification with OpenCV

This Python project performs feature-based registration and identification of grayscale images using OpenCV. It is designed to detect and compare feature points between two sets of images (e.g., two inspection passes of industrial components), identify matching pairs, and log results and processing time.

## Features

- ROI-based image preprocessing
- Binarization and ORB feature point extraction
- Adaptive Non-Maximal Suppression (ANMS) for uniform keypoint distribution
- Image descriptor generation and storage
- Two identification algorithms: Nearest Neighbor (NN) and Descriptor Distance (DIS)
- Configurable parameters via `.ini` file
- Detailed runtime logging and performance statistics

## Folder Structure

```
project_root/
│
├── imgs/
│   ├── Durchgang1/        # First image dataset (registration set)
│   └── Durchgang2/        # Second image dataset (identification set)
├── res/
│   ├── bin/               # Binarized images
│   ├── roi/               # Region-of-interest cropped images
│   ├── featpts/           # Feature points (raw ORB)
│   ├── featpts-anms/      # Feature points after ANMS
│   └── files/             # Feature point data files
├── config.ini             # Configuration file
├── main.py                # Main execution script
└── cvfunc.py              # Image processing and analysis functions
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy

Install required packages using:

```bash
pip install opencv-python numpy
```

## Usage

1. **Prepare your image datasets** in `imgs/Durchgang1/` and `imgs/Durchgang2/`.
2. **Configure processing settings** in `config.ini` (see below).
3. **Run the script**:

```bash
python main.py
```

The script will:
- Preprocess the images,
- Extract and store feature points,
- Compare each image from the first dataset with those from the second,
- Determine the best match based on the selected algorithm,
- Log all key information in a results file under `res/`.

## Configuration (`config.ini`)

```ini
# paramters for computer vision functions
# algor can be "nn" or "dis"
[CV_PARAMS]
algor = dis
row1 = 800
row2 = 1600
col1 = 800
col2 = 1900
thresh = 190
nfeatures = 110
num_to_keep = 15
min_rad = 30

# params for dis algorithm
max_diff = 3

# part index in folder, not list index
# start counting from 0, including this part
# Enter "fst_part = 0" and "lst_part = ALL" to run script for all parts
# register = 0 --> False, register = 1 --> True
[EXEC_PARAMS]
fst_part = 0
lst_part = 2
register = 1

```

## Example Output

Console:
```
image_001.png has the best match with image_001.png. Corresp points: 28
image_002.png has the best match with image_003.png. Corresp points: 22
...
3 from 5 parts successfully identified
Identification runtime: 2.34 sec
```

Log file (`res/log_nn.txt`):
```
#### computer vision parameters ####
algorithm = nn
roi (row1,row2,col1,col2) = 100,400,50,300
...
Total runtime: 5.67 sec
...
```

## License

This project is released under the MIT License.

## Acknowledgements

Developed using OpenCV’s ORB feature detection and custom preprocessing tools. For questions or contributions, feel free to open an issue or pull request.
