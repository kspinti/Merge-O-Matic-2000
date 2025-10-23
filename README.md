# 📊 Merge-O-Matic 2000

A lightweight, python-based data combination tool that allows you to upload multiple CSV or Excel files, preview their data, select columns to include, visualize results interactively, and automatically generate a combined Excel analysis file — all from a simple browser interface.

---

## 🚀 Features

✅ **Smart file upload and preview**
- Supports `.csv` and `.xlsx` files.
- Automatically detects and skips non-data header lines (e.g., notes or metadata before the real header).
- Displays column names and a short data preview for each file.

✅ **Flexible column selection**
- Choose which columns from each uploaded file to include in the combined output.
- Assign new names and units for clarity.

✅ **Interactive graphing**
- Load all selected datasets into a single, shared Plotly graph.
- Instantly toggle datasets on/off for fast visualization.
- Unified time axis for easy comparison.

✅ **Automatic Excel export**
- Combines selected datasets into one analysis-ready Excel file.
- Includes consistent formatting and date handling.
- One-click download!

---

## 🧠 How It Works

1. **Upload your files**
   - Drag and drop multiple `.csv` or `.xlsx` files into the uploader.
   - The app automatically detects headers and previews the first few rows.

2. **Select your data**
   - For each file, choose which columns to include in the combined dataset.
   - Optionally rename or assign units.

3. **Visualize your data (optional)**
   - Enable graphing to load all selected datasets into memory.
   - Pick which datasets to display on an interactive Plotly graph.
   - Zoom, pan, and hover for details.

4. **Combine and download**
   - Click **“Create combined data file”** to generate an analysis-ready Excel file.
   - Download the file directly from your browser.
