import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import warnings
import os
import gc
import json
from pathlib import Path
from tkinter import filedialog
from colorama import init, Fore, Back, Style

import platform


class ConfigManager:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path='default_config.json'):
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found. Using default values.")
            return False
        return True
    
    def get(self, *keys):
        current = self._config
        for key in keys:
            if current is not None:
                current = current.get(key)
        return current
    # Add this method to the ConfigManager class
    def _save_current_config(self):
        """Save the current configuration back to the file"""
        try:
            with open('current_config.json', 'w') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save current configuration: {e}")

def get_system_info():
    """Get detailed system information for adaptive configuration"""
    info = {
        'total_ram': 0,
        'available_ram': 0,
        'cpu_count': os.cpu_count() or 1,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_memory': 0,
        'platform': platform.system(),
        'is_64bit': platform.machine().endswith('64')
    }
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_ram'] = mem.total / (1024 ** 3)  # GB
        info['available_ram'] = mem.available / (1024 ** 3)  # GB
    except ImportError:
        print("psutil not installed. RAM information will not be available.")
    
    if info['cuda_available']:
        try:
            info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        except Exception as e:
            print(f"Error getting CUDA memory info: {e}")
    
    return info

def calculate_optimal_settings(system_info):
    """Calculate optimal settings based on system capabilities"""
    settings = {}
    
    # Calculate optimal chunk size based on available memory
    if system_info['cuda_available']:
        # Use 70% of available CUDA memory for processing
        available_memory = system_info['cuda_memory'] * 0.7
        settings['chunk_height'] = min(2000, int(available_memory * 200))  # Rough estimation
    else:
        # Use 50% of available RAM for processing
        available_memory = system_info['available_ram'] * 0.5
        settings['chunk_height'] = min(1000, int(available_memory * 100))
    
    # Adjust tile size based on available memory
    settings['tile_size'] = min(400, settings['chunk_height'] // 2)
    
    # Calculate optimal CUDA settings
    if system_info['cuda_available']:
        # Adjust split size based on available CUDA memory
        split_size = min(128, int(system_info['cuda_memory'] * 1024 * 0.3))  # 30% of CUDA memory
        settings['cuda_alloc_conf'] = f'max_split_size_mb:{split_size}'
    
    return settings

def initialize_system():
    """Initialize system with adaptive configuration based on hardware capabilities"""
    config = ConfigManager()
    
    try:
        # Get system information
        system_info = get_system_info()
        
        # Calculate optimal settings
        optimal_settings = calculate_optimal_settings(system_info)
        
        # Update configuration with optimal settings
        if not config._config:
            config._config = {}
        
        # Update processing settings
        if 'processing' not in config._config:
            config._config['processing'] = {}
        config._config['processing']['chunk_height'] = optimal_settings['chunk_height']
        config._config['processing']['tile_size'] = optimal_settings['tile_size']
        
        # Initialize CUDA if available
        if system_info['cuda_available']:
            if config.get('cuda', 'empty_cache_on_start'):
                torch.cuda.empty_cache()
            
            # Set optimal CUDA allocation configuration
            cuda_conf = optimal_settings.get('cuda_alloc_conf') or config.get('cuda', 'alloc_conf')
            if cuda_conf:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cuda_conf
                
            # Set device to GPU
            torch.cuda.set_device(0)
            
            # Enable autograd anomaly detection in debug mode
            if config.get('debug', 'enabled'):
                torch.autograd.set_detect_anomaly(True)
        else:
            print("CUDA not available. Using CPU for processing (this will be slower)")
        
        # Configure image processing
        if config.get('image', 'max_pixels') is not None:
            Image.MAX_IMAGE_PIXELS = config.get('image', 'max_pixels')
        else:
            # Set based on available memory
            max_pixels = int(system_info['available_ram'] * 1024 * 1024 * 0.3)  # 30% of available RAM
            Image.MAX_IMAGE_PIXELS = max_pixels
        
        # Configure warnings
        if not config.get('debug', 'enabled'):
            warnings.filterwarnings("ignore")
        
        # Save optimized configuration
        config._save_current_config()
        
        print("\nSystem initialized with the following configuration:")
        print(f"Processing chunk height: {optimal_settings['chunk_height']}")
        print(f"Tile size: {optimal_settings['tile_size']}")
        if system_info['cuda_available']:
            print(f"CUDA configuration: {cuda_conf}")
        print(f"Maximum image pixels: {Image.MAX_IMAGE_PIXELS:,}")
        
        return True
        
    except Exception as e:
        print(f"Error during system initialization: {e}")
        if config.get('debug', 'enabled'):
            import traceback
            traceback.print_exc()
        return False



def validate_image_file(image_path):
    """
    Validates that the image file exists and can be opened.
    Returns the image if valid, None if not.
    """
    config = ConfigManager()
    
    if image_path[0] == '"':
        image_path = image_path.replace('"', "") 
        print(image_path)
    try:
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' does not exist.")
            return None
            
        if not os.path.isfile(image_path):
            print(f"Error: '{image_path}' is not a file.")
            return None
            
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Error: File '{image_path}' is empty.")
            return None
            
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img
        
    except UnidentifiedImageError:
        supported_formats = config.get('image', 'supported_formats')
        print(f"Error: Could not identify image format of '{image_path}'")
        print("Supported formats:", ', '.join(supported_formats))
        return None
    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        return None

def check_cuda():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA not available")

def get_memory_info():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - allocated - reserved
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'free': free
        }
    return None

def process_image_in_chunks(image_path, output_path):
    config = ConfigManager()
    model_path = config.get('processing', 'model', 'path')
    chunk_height = config.get('processing', 'chunk_height')
    tile_size = config.get('processing', 'tile_size')
    overlap = config.get('processing', 'overlap')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        return False
        
    img = validate_image_file(image_path)
    if img is None:
        return False
    
    if config.get('output', 'create_dirs'):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model
        state_dict = torch.load(model_path, map_location=device)['params_ema']
        model_config = config.get('processing', 'model')
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=model_config['num_feat'],
            num_block=model_config['num_block'],
            num_grow_ch=model_config['num_grow_ch'],
            scale=model_config['scale']
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.half()
        
        # Create upsampler
        upsampler = RealESRGANer(
            scale=model_config['scale'],
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=model_config['tile_pad'],
            pre_pad=model_config['pre_pad'],
            half=True if device.type == 'cuda' else False,
            device=device
        )
        
        width, height = img.size
        output_height = height * model_config['scale']
        output_width = width * model_config['scale']
        output_img = Image.new('RGB', (output_width, output_height))
        
        for start_y in range(0, height, chunk_height - overlap):
            torch.cuda.empty_cache()
            gc.collect()
            
            end_y = min(start_y + chunk_height, height)
            
            if end_y < height:
                end_y += overlap
            
            print(f"Processing chunk: {start_y} -> {end_y} of {height}")
            
            chunk = img.crop((0, start_y, width, end_y))
            chunk_np = np.array(chunk)
            
            try:
                output_chunk, _ = upsampler.enhance(chunk_np, outscale=model_config['scale'])
                output_chunk_img = Image.fromarray(output_chunk)
                
                paste_y = start_y * model_config['scale']
                
                if start_y > 0:
                    paste_y += overlap * 2
                    output_chunk_img = output_chunk_img.crop((0, overlap * 4, output_chunk_img.width, output_chunk_img.height))
                
                output_img.paste(output_chunk_img, (0, paste_y))
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"Memory error processing chunk: {str(e)}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        try:
            output_img.save(output_path)
            print(f"Image successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    Ascii_art = """ █████  ████████████████   █████████    █████████   █████████  █████      ██████████ ███████████  
░░███  ░░███░░███░░░░░███ ███░░░░░███  ███░░░░░███ ███░░░░░███░░███      ░░███░░░░░█░░███░░░░░███ 
 ░███   ░███ ░███    ░███░███    ░░░  ███     ░░░ ░███    ░███ ░███       ░███  █ ░  ░███    ░███ 
 ░███   ░███ ░██████████ ░░█████████ ░███         ░███████████ ░███       ░██████    ░██████████  
 ░███   ░███ ░███░░░░░░   ░░░░░░░░███░███         ░███░░░░░███ ░███       ░███░░█    ░███░░░░░███ 
 ░███   ░███ ░███         ███    ░███░░███     ███░███    ░███ ░███      █░███ ░   █ ░███    ░███ 
 ░░████████  █████       ░░█████████  ░░█████████ █████   ██████████████████████████ █████   █████
  ░░░░░░░░  ░░░░░         ░░░░░░░░░    ░░░░░░░░░ ░░░░░   ░░░░░░░░░░░░░░░░░░░░░░░░░░ ░░░░░   ░░░░░ """
    print(f"\n\n{Fore.CYAN}{Style.BRIGHT}{Ascii_art}{Style.RESET_ALL}\n\n")

    initialize_system()
    
    print('Select the image to process:')

    config = ConfigManager()
    supported_formats = config.get('image', 'supported_formats')
    input_image = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", " ".join(f"*{fmt}" for fmt in supported_formats))]
    )
    out_path = filedialog.asksaveasfilename(
        title="Save Image As",
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png"),
            ("JPEG", "*.jpg"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tif")
        ]
    )

    check_cuda()
    memory_info = get_memory_info()
    if memory_info:
        print("\nMemory Status:")
        print(f"Total: {memory_info['total']:.2f} GB")
        print(f"Free: {memory_info['free']:.2f} GB")
        print(f"Allocated: {memory_info['allocated']:.2f} GB")
        print(f"Reserved: {memory_info['reserved']:.2f} GB")

    success = process_image_in_chunks(
        image_path=input_image,
        output_path=out_path
    )
    
    if success:
        print("Processing completed successfully")
    else:
        print("Processing failed")