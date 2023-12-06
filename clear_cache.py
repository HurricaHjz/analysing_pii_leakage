import torch

# Print the current memory usage.
def print_memory_usage():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated / 1024 ** 3:.2f} GB")
    print(f"Cached memory: {cached / 1024 ** 3:.2f} GB")

print("Before clearing cache:")
print_memory_usage()

# Clear cache
torch.cuda.empty_cache()

print("\nAfter clearing cache:")
print_memory_usage()
