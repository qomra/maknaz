import os
import shutil
import argparse

def copy_symlink_files(src_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for item in os.listdir(src_directory):
        item_path = os.path.join(src_directory, item)
        if os.path.islink(item_path):
            actual_file = os.readlink(item_path)
            actual_file_path = os.path.join(os.path.dirname(item_path), actual_file)
            target_file_path = os.path.join(target_directory, item)
            
            # Copy the actual file to the target directory with the symlink name
            shutil.copy2(actual_file_path, target_file_path)
            print(f"Copied {actual_file_path} to {target_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Copy actual files from symbolic links to a target directory.')
    parser.add_argument('src_directory', type=str, help='The source directory containing the symbolic links')
    parser.add_argument('target_directory', type=str, help='The target directory to copy the actual files to')

    args = parser.parse_args()

    copy_symlink_files(args.src_directory, args.target_directory)

if __name__ == "__main__":
    main()
