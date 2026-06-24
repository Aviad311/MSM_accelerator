import os
import re
import sys
import argparse
from pathlib import Path

def remove_module(module_name, target_dir="."):
    target_path = Path(target_dir)
    sv_file = target_path / f"{module_name}.sv"
    
    # 1. Delete the file named 'X.sv'
    if sv_file.exists():
        print(f"Deleting {sv_file}...")
        sv_file.unlink()
    else:
        print(f"Warning: {sv_file} not found.")

    # Patterns
    # Matches `include "X.sv"` or `include 'X.sv'`
    include_pattern = re.compile(rf'^\s*`include\s+["\']{module_name}\.sv["\'].*$', re.MULTILINE)
    
    # Matches module instantiations: 
    # ModuleName [#(params)] InstanceName (ports);
    # Handles multi-line blocks by matching until the final );
    instantiation_pattern = re.compile(
        rf'\b{module_name}\b\s+(?:#\s*\([\s\S]*?\)\s*)?\w+\s*\([\s\S]*?\)\s*;',
        re.MULTILINE
    )

    # 2. Scan all .v and .sv files
    for file_path in target_path.glob("**/*.[vs]v"):
        if file_path == sv_file:
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()

        original_content = content
        
        # 3. Remove include statements
        content = include_pattern.sub("", content)
        
        # 4. Remove instantiations
        content = instantiation_pattern.sub("", content)
        
        # Clean up potential double newlines left behind
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        if content != original_content:
            print(f"Updating {file_path}...")
            with open(file_path, 'w') as f:
                f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanly remove a Verilog/SystemVerilog module.")
    parser.add_argument("module", help="Name of the module to remove (without .sv extension)")
    parser.add_argument("--dir", default=".", help="Directory to scan (default: current)")
    
    args = parser.parse_args()
    remove_module(args.module, args.dir)

