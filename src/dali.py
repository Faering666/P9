import torch
import numpy as np
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# ==========================================
# 1. Custom Data Source Adapter
# ==========================================
class ExternalSourceAdapter:
    """
    Wraps a legacy PyTorch Dataset to feed DALI's ExternalSource.
    Ensures data is returned as raw bytes (uint8) for GPU decoding.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.epoch_idx = 0

    def __call__(self, sample_info):
        if sample_info.idx_in_epoch >= len(self.dataset):
            raise StopIteration()
        
        # Shuffle at the start of each epoch
        if self.shuffle and sample_info.idx_in_epoch == 0:
            rng = np.random.default_rng(seed=42 + self.epoch_idx)
            rng.shuffle(self.indices)
            self.epoch_idx += 1
            
        # Retrieve real index
        real_idx = self.indices[sample_info.idx_in_epoch]
        
        # Fetch data from legacy dataset
        # Note: This assumes the legacy dataset __getitem__ can be modified 
        # or bypassed to return raw bytes. If not, we might load the path here.
        # Hypothetical access pattern:
        raw_data, label = self.dataset[real_idx] 
        
        # Ensure numpy format for DALI
        # raw_data should be a uint8 array (encoded JPEG)
        return np.array(raw_data, dtype=np.uint8), np.array([label], dtype=np.int64)

# ==========================================
# 2. DALI Pipeline Definition
# ==========================================
@pipeline_def
def create_training_pipeline(source_callable, crop_size=256):
    # Data Ingestion
    images, labels = fn.external_source(
        source=source_callable,
        num_outputs=2,
        dtype=torch.float16,
        parallel=True,       # Enable multiprocessing
        batch=False          # Source returns single samples
    )
    
    # GPU-Accelerated Decoding & Geometric Transform
    # Fused Decode + Random Resized Crop
    images = fn.decoders.image_random_crop(
        images,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[0.8, 1.25],
        random_area=[0.1, 1.0],
        num_attempts=100
    )
    
    # Resize to fixed dimensions
    images = fn.resize(
        images, 
        size=[crop_size, crop_size],
        interp_type=types.INTERP_LINEAR
    )
    
    # Color Augmentation (GPU)
    # Random Brightness and Contrast
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=[0.8, 1.2]),
        contrast=fn.random.uniform(range=[0.8, 1.2])
    )
    
    # Horizontal Flip
    mirror = fn.random.coin_flip(probability=0.5)

    # Normalization & Layout Conversion (HWC -> CHW)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mirror=mirror
    )
    
    # Ensure labels are on GPU
    labels = labels.gpu()
    
    return images, labels

# ==========================================
# 3. Drop-In DataLoader Replacement
# ==========================================
class DALILoader:
    def __init__(self, dataset, batch_size, num_threads=4, device_id=0):
        # Initialize Source
        self.source = ExternalSourceAdapter(dataset, batch_size, shuffle=True)
        
        # Initialize Pipeline
        self.pipe = create_training_pipeline(
            source_callable=self.source,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id
        )
        self.pipe.build()
        
        # Initialize Iterator
        self.iterator = DALIGenericIterator(
            [self.pipe],
            ["data", "label"],
            size=-1, # Auto-detect size via StopIteration
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP
        )
        
    def __len__(self):
        return len(self.source.dataset) // self.source.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
            # Reformat DALI output to match PyTorch (Tensor, Tensor)
            return data["data"], data["label"].squeeze()
        except StopIteration:
            raise
