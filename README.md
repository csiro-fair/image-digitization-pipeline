# Image Digitization Pipeline

A Marimba Pipeline for processing and standardizing historical slide-film imagery collections from marine ecological 
surveys. The Pipeline specializes in reconstructing spatio-temporal context from historical records while preserving 
image quality and experimental metadata.


## Overview

The Image Digitization Pipeline is designed to process historical slide-film imagery collected during CSIRO marine 
monitoring campaigns between 1993 and 1996. It handles data from the CSIRO Osprey Camera System (OCS) and Towed Camera 
Array (TACOS), which captured benthic habitat imagery across the continental shelf.

Key capabilities include:

- Automated processing of digitized slide-film photographs
- Spatio-temporal interpolation for reconstructing image positions
- Integration of survey metadata including GPS coordinates, depths, and timestamps
- Quality-controlled image rotation and sequence reconstruction
- Generation of FAIR-compliant datasets with embedded metadata


## Requirements

The Image Digitization Pipeline is built on the [Marimba](https://github.com/csiro-fair/marimba) framework which 
includes most necessary dependencies for this Pipeline. Additional requirements include:
- openpyxl


## Installation

Create a new Marimba project and add the Image Digitization Pipeline:

```bash
marimba new project my-digitization-project
cd my-digitization-project
marimba new pipeline image_digitization https://github.com/csiro-fair/image-digitization-pipeline.git
```


## Configuration

### Pipeline Configuration
The Pipeline uses a configuration-free approach, with all necessary parameters derived from input data.

### Collection Configuration
Each Collection requires:
- `batch_data_path`: Path to batch processing metadata CSV
- `inventory_data_path`: Path to film inventory Excel file
- `import_path`: Base path for source image directories


## Usage

### Importing Data

Import collections with batch-specific configurations:
```bash
marimba import batch1a /path/to/source/batch1a/* --operation link \
--config '{
    "batch_data_path": "/path/to/batch1a.csv",
    "inventory_data_path": "/path/to/inventory.xlsx",
    "import_path": "/path/to/source/batch1a/"
}'
```

For large digitization projects, multiple batches can be processed:
```bash
# Import multiple batches
marimba import batch1a /path/to/batch1a/* --config '...'
marimba import batch1b /path/to/batch1b/* --config '...'
marimba import batch1c /path/to/batch1c/* --config '...'
```

### Source Data Structure

The Pipeline expects digitized slide-film data organized by sequential folders:
```
source/
└── batch1a/
    ├── 00000001/
    │   └── *.jpg       # Digitized slide-film images
    ├── 00000002/
    │   └── *.jpg
    └── ...
```

### Processing

```bash
marimba process
```

During processing, the Image Digitization Pipeline:
1. Creates a hierarchical directory structure by survey, platform, and deployment
2. Interpolates geographic coordinates between deployment start/end points
3. Rotates and sequences images based on film orientation
4. Integrates station metadata (coordinates, depth, timestamp)
5. Generates thumbnail overviews for visual validation

### Packaging

```bash
marimba package my-digitized-dataset \
--operation link \
--version 1.0 \
--contact-name "Keiko Abe" \
--contact-email "keiko.abe@email.com"
```

The `--operation link` flag creates hard links instead of copying files, optimizing storage for large datasets.


## Processed Data Structure

```
SEFHES/                                         # Root dataset directory
├── data/                                       # Directory containing all processed data
│   └── image_digitization/                     # Image Digitization Pipeline data directory
│       └── [Survey_ID]/                        # Survey directories (SS199305, etc.)
│           └── [Platform_ID]/                  # Platform directories (OCS, TACOS)
│               └── [Survey_ID]_[Station]/      # Station directories
│                   ├── data/                   # Station metadata
│                   │   └── *.CSV               # Navigation and measurement data
│                   ├── stills/                 # Processed images
│                   │   └── *.JPG               # Individual images
│                   ├── thumbnails/             # Thumbnail images
│                   │   └── *.JPG               # Individual thumbnail images
│                   └── *_OVERVIEW.JPG          # Overview image
├── logs/                                       # Directory containing all processing logs
│   ├── pipelines/                              # Pipeline-specific logs
│   │   └── image_digitization.log              # Logs from Image Digitization Pipeline
│   ├── dataset.log                             # Dataset packaging logs
│   └── project.log                             # Overall project processing logs
├── pipelines/                                  # Directory containing pipeline code
│   └── image_digitization/                     # Pipeline-specific directory
│       ├── repo/                               # Pipeline source code repository
│       │   ├── image_digitization.pipeline.py  # Pipeline implementation
│       │   ├── LICENSE                         # Pipeline license file
│       │   └── README.md                       # Pipeline README file
│       └── pipeline.yml                        # Pipeline configuration
├── ifdo.yml                                    # Dataset-level iFDO metadata file
├── manifest.txt                                # File manifest with SHA256 hashes
├── map.png                                     # Spatial visualization of dataset
└── summary.md                                  # Dataset summary and statistics
```


## Metadata

The Pipeline captures comprehensive metadata including:

### Survey Metadata
- GPS coordinates
- Sampling depths
- Collection times
- Platform configurations
- Environmental parameters

### Technical Metadata
- Image acquisition parameters
- Camera configurations
- Processing parameters
- Quality metrics

### Image-Specific Data
- Spatial coordinates (interpolated)
- Temporal information
- Platform details
- Deployment context

All metadata is standardized using the iFDO schema (v2.1.0) and embedded in both image EXIF tags and dataset-level files.


## Contributors

The Image Digitization Pipeline was developed by:
- Christopher Jackett (CSIRO)
- Candice Untiedt (CSIRO)
- Franziska Althaus (CSIRO)
- David Webb (CSIRO)
- Ben Scoulding (CSIRO)


## License

The Image Digitization Pipeline is distributed under the [CSIRO BSD/MIT](LICENSE) license.


## Contact

For inquiries related to this repository, please contact:

- **Chris Jackett**  
  *Software Engineer, CSIRO*  
  Email: [chris.jackett@csiro.au](mailto:chris.jackett@csiro.au)
