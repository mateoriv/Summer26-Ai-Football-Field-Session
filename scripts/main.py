#!/usr/bin/env python3
"""
Main script for AI Football Field Session Analysis
Coordinates all detection and analysis modules
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Main function to coordinate all analysis modules"""
    parser = argparse.ArgumentParser(description='AI Football Field Session Analysis')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache', help='Output directory for results')
    parser.add_argument('--modules', nargs='+', 
                       choices=['player', 'yard_line', 'yard_marker', 'homography'],
                       default=['player', 'yard_line', 'yard_marker'],
                       help='Analysis modules to run')
    
    args = parser.parse_args()
    
    # Validate input video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting analysis of: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Modules to run: {args.modules}")
    
    # TODO: Implement module coordination
    # This will call the appropriate detection modules based on args.modules
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()