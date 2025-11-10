# Installation Guide

## Quick Fix for Python 3.12 Compatibility

The error you encountered is due to outdated package versions in `requirements.txt` that aren't compatible with Python 3.12.

### Solution 1: Automatic Installation (Recommended)

I've updated `requirements.txt` to use the latest compatible versions. Now try:

```bash
# First, upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# Then install dependencies
pip install -r requirements.txt
```

### Solution 2: Use the Batch Script (Windows)

If you're on Windows, you can use the provided batch file:

```bash
install.bat
```

### Solution 3: Manual Step-by-Step

If you still encounter issues, install packages one by one:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install Flask
pip install numpy
pip install pandas
pip install scikit-learn
pip install plotly
pip install Werkzeug
```

### Solution 4: Use a Virtual Environment (Recommended for Clean Setup)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## After Installation

Once installed, run the application:

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

## Troubleshooting

### If you get "setuptools.build_meta" error:
```bash
python -m pip install --upgrade setuptools wheel
```

### If numpy fails to install:
```bash
pip install numpy --upgrade
```

### If you're on Windows and get permission errors:
- Run PowerShell/CMD as Administrator, or
- Use the `--user` flag: `pip install -r requirements.txt --user`

## Notes

- The updated `requirements.txt` now uses the latest compatible versions instead of pinned versions
- This ensures compatibility with Python 3.12
- All packages will automatically resolve to compatible versions

