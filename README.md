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
