# P9

## Generating database patches for training
Generate dataset patches from sri lanka dataset images
```bash
python3 utils/generate_dataset.py --data_path data/sri_lanka/
```

Remove dataset patches (not original images)
```bash
python3 utils/remove_dataset.py --data_path data/sri_lanka/ [--dry-run]
```

## Align dataset
As an example with the Sri Lanka dataset in the path `data/sri_lanka` we can 
use the tool to align them. By default they will align by the green ms image.
To use, run the command:
```bash
python utils/align_images.py --input_folder=/path/to/dataset [--output_folder=/path/to/save/aligned_images]
```
By default the aligned images will be saved to data/aligned_images if nothing else is specified.