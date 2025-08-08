import os
import subprocess

# Optional: install pipreqs if not already installed
try:
    import pipreqs
except ImportError:
    subprocess.run(["pip", "install", "pipreqs"])

def generate_minimal_requirements(project_path="."):
    print(f"Scanning project at: {os.path.abspath(project_path)}")
    subprocess.run(["pipreqs", project_path, "--force"])
    print("✅ Minimal requirements.txt generated.")

if __name__ == "__main__":
    generate_minimal_requirements()