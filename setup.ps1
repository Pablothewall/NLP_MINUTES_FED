# Create a virtual environment
python -m venv venv

# Activate the virtual environment
. .\venv\Scripts\Activate

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install requirements
python -m pip install -r requirements.txt

# Display activation instructions
Write-Host "Virtual environment created and activated. To deactivate the virtual environment, run: deactivate"

# Create .gitignore file
New-Item -Type File -Name .gitignore

# Add common Python and virtual environment exceptions to .gitignore
@(
    "# .gitignore file for Python projects",
    "venv/"
) | Add-Content .gitignore