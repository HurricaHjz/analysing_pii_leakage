from dataclasses import dataclass


@dataclass
class LocalConfigs:
    CACHE_DIR = "./cache_memory"
    IMAGENET_ROOT = "./img_root"