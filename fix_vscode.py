import json
import os

settings_path = os.path.expandvars(r"%APPDATA%\Code\User\settings.json")

# Load existing settings
if os.path.exists(settings_path):
    with open(settings_path, "r") as f:
        settings = json.load(f)
else:
    settings = {}

# Set the correct Python interpreter so Pylance finds all installed packages
settings["python.defaultInterpreterPath"] = r"C:\Users\rudra\AppData\Local\Programs\Python\Python310\python.exe"
settings["python.analysis.typeCheckingMode"] = "basic"
settings["pyre2.enabled"] = False

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=4)

print("VS Code settings updated successfully!")
print("Please restart VS Code now.")
