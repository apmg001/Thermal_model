# diagnostic.py
import sys
import os
import subprocess

print("=== Python Diagnostic ===")
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"\nSite packages:")

# Check site packages
import site
for path in site.getsitepackages():
    print(f"  {path}")
    if os.path.exists(path):
        for item in os.listdir(path):
            if 'bm3d' in item.lower() or 'bm4d' in item.lower():
                print(f"    Found: {item}")

print(f"\nPATH: {os.environ.get('PATH', '')}")

# Try to compile a simple Fortran test
print("\n=== Fortran Compiler Check ===")
try:
    result = subprocess.run(['gfortran', '--version'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Fortran compiler found")
        print(result.stdout[:100])
    else:
        print("✗ Fortran compiler not found")
except:
    print("✗ Fortran compiler not installed")