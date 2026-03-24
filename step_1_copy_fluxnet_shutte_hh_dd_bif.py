import os
import zipfile
import shutil
import re
from pathlib import Path

def setup_directories():
    """Create output directories"""
    directories = [
        './fluxnet_data/HH',
        './fluxnet_data/DD',
        './fluxnet_data/BIF'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def is_target_file(filename, file_type):
    """
    Check whether a file is a target file
    
    Args:
        filename: file name
        file_type: file type ('HH', 'DD', or 'BIF')
    
    Returns:
        bool: True if target file
    """
    if file_type in ['HH', 'DD']:
        # Match FLUXMET files only
        pattern = f'FLUXNET_FLUXMET_{file_type}_'
        return pattern in filename and filename.endswith('.csv')
    
    elif file_type == 'BIF':
        # BIF file: contains BIF_ but excludes BIFVARINFO
        return 'BIF_' in filename and 'BIFVARINFO' not in filename and filename.endswith('.csv')
    
    return False

def extract_fluxnet_files(zip_path, output_dirs):
    """
    Extract required FLUXNET files from ZIP archive
    
    Args:
        zip_path: path to ZIP file
        output_dirs: dict containing 'HH', 'DD', 'BIF' directories
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Extract site name
            site_name = None
            for filename in file_list:
                if 'BIF_' in filename and 'BIFVARINFO' not in filename:
                    site_name = filename.split('_BIF_')[0]
                    break
            
            if not site_name:
                print(f"Warning: No BIF file found in {zip_path}, skipping")
                return
            
            print(f"\nProcessing site: {site_name}")
            
            found_files = {'HH': 0, 'DD': 0, 'BIF': 0}
            
            for filename in file_list:
                target_dir = None
                file_category = None
                
                # HH files
                if is_target_file(filename, 'HH'):
                    target_dir = output_dirs['HH']
                    file_category = 'HH'
                    print(f"  Found FLUXMET_HH file: {filename}")
                    
                # DD files
                elif is_target_file(filename, 'DD'):
                    target_dir = output_dirs['DD']
                    file_category = 'DD'
                    print(f"  Found FLUXMET_DD file: {filename}")
                    
                # BIF files
                elif is_target_file(filename, 'BIF'):
                    target_dir = output_dirs['BIF']
                    file_category = 'BIF'
                    print(f"  Found BIF file: {filename}")
                
                if target_dir is None:
                    continue
                
                target_path = os.path.join(target_dir, filename)
                
                if os.path.exists(target_path):
                    print(f"    File already exists, skipping: {filename}")
                    continue
                
                try:
                    source = zip_ref.open(filename)
                    with open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    print(f"    Extracted: {filename}")
                    found_files[file_category] += 1
                except Exception as e:
                    print(f"    Failed to extract {filename}: {e}")
            
            total_found = sum(found_files.values())
            if total_found > 0:
                print(f"  Site {site_name} extraction complete: HH={found_files['HH']}, DD={found_files['DD']}, BIF={found_files['BIF']}")
            else:
                print(f"  Warning: No target files found for site {site_name}")
                    
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file")
    except Exception as e:
        print(f"Error processing {zip_path}: {e}")

def verify_extracted_files(output_dirs):
    """Verify extracted files"""
    print("\n" + "="*50)
    print("Verifying extracted files:")
    
    for dir_name, dir_path in output_dirs.items():
        if not os.path.exists(dir_path):
            continue
            
        files = os.listdir(dir_path)
        pattern_map = {
            'HH': 'FLUXMET_HH_',
            'DD': 'FLUXMET_DD_',
            'BIF': 'BIF_'
        }
        
        expected_pattern = pattern_map.get(dir_name, '')
        
        valid_files = [f for f in files if expected_pattern in f and f.endswith('.csv')]
        invalid_files = [f for f in files if f not in valid_files]
        
        print(f"\n{dir_path}:")
        print(f"  Total files: {len(files)}")
        print(f"  Valid files: {len(valid_files)}")
        
        if invalid_files:
            print(f"  Warning: {len(invalid_files)} invalid files found:")
            for f in invalid_files[:5]:
                print(f"    - {f}")
        
        if valid_files:
            print(f"  Example valid files: {valid_files[:3]}")

def main():
    """Main function"""
    file_path = '../../Data/FLUXNET_shuttle'
    
    output_dirs = {
        'HH': './fluxnet_data/HH',
        'DD': './fluxnet_data/DD',
        'BIF': './fluxnet_data/BIF'
    }
    
    print("Creating output directories...")
    setup_directories()
    
    if not os.path.exists(file_path):
        print(f"Error: source directory {file_path} does not exist")
        return
    
    zip_files = [f for f in os.listdir(file_path) if f.endswith('.zip')]
    
    if not zip_files:
        print(f"No ZIP files found in {file_path}")
        return
    
    print(f"Found {len(zip_files)} ZIP files")
    print("\nStart extraction...")
    print("Note: only extracting FLUXMET_HH, FLUXMET_DD, and BIF files")
    
    total_processed = 0
    
    for zip_file in sorted(zip_files):
        zip_path = os.path.join(file_path, zip_file)
        print(f"\n[{total_processed + 1}/{len(zip_files)}] Processing: {zip_file}")
        extract_fluxnet_files(zip_path, output_dirs)
        total_processed += 1
    
    print("\n" + "="*50)
    print("Extraction completed! Summary:")
    
    total_files = 0
    for dir_name, dir_path in output_dirs.items():
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            file_count = len(files)
            total_files += file_count
            
            if dir_name == 'HH':
                fluxmet_files = [f for f in files if 'FLUXMET_HH' in f]
                print(f"\n{dir_path} (hourly data):")
                print(f"  Total files: {file_count}")
                print(f"  FLUXMET_HH files: {len(fluxmet_files)}")
                
            elif dir_name == 'DD':
                fluxmet_files = [f for f in files if 'FLUXMET_DD' in f]
                print(f"\n{dir_path} (daily data):")
                print(f"  Total files: {file_count}")
                print(f"  FLUXMET_DD files: {len(fluxmet_files)}")
                
            elif dir_name == 'BIF':
                bif_files = [f for f in files if 'BIF' in f and 'BIFVARINFO' not in f]
                print(f"\n{dir_path} (BIF metadata):")
                print(f"  Total files: {file_count}")
                print(f"  BIF files: {len(bif_files)}")
    
    print(f"\nTotal extracted files: {total_files}")
    
    verify_extracted_files(output_dirs)

if __name__ == "__main__":
    main()