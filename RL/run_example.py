#!/usr/bin/env python3
"""
Example launcher script for easy execution of RL examples.
"""

import sys
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run RL examples')
    parser.add_argument('example', choices=['distributed', 'single', 'priority', 'r2d2', 'list'], 
                       help='Example to run or "list" to see available examples')
    parser.add_argument('--args', type=str, default='', 
                       help='Additional arguments to pass to the example script')
    
    args = parser.parse_args()
    
    if args.example == 'list':
        print("Available examples:")
        print("  distributed - Run distributed Space Invaders DQN example")
        print("  single      - Run single-threaded Space Invaders DQN example")
        print("  priority    - Prioritized vs uniform replay buffer examples")
        print("  r2d2        - R2D2 (LSTM + sequence replay) example")
        print("\nExample usage:")
        print("  python run_example.py distributed --args '--mode test'")
        print("  python run_example.py single --args '--test-only'")
        print("  python run_example.py priority --args '--mode test'")
        print("  python run_example.py r2d2 --args '--mode fast --episodes 5'")
        print("  python run_example.py r2d2 --args '--mode compare --episodes 3'")
        print("  python run_example.py distributed --args '--mode train --episodes 100 --workers 4'")
        return 0
    
    # Map example names to scripts
    example_scripts = {
        'distributed': 'examples/distributed_space_invaders_example.py',
        'single': 'examples/single_threaded_example.py',
        'priority': 'examples/priority_replay_example_usage.py',
        'r2d2': 'examples/r2d2_example.py'
    }
    
    script = example_scripts[args.example]
    
    if not os.path.exists(script):
        print(f"Error: Example script {script} not found!")
        return 1
    
    # Prepare command
    cmd = [sys.executable, script]
    if args.args:
        cmd.extend(args.args.split())
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    # Execute the example
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running example: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())