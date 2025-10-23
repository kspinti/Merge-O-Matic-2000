import streamlit as st
import pandas as pd
import io
import openpyxl
from datetime import datetime
from copy import copy
import os
import plotly.express as px
from io import BytesIO

def read_flexible_csv(file_bytes, max_check_lines=30):
    """
    Read a CSV file that may have extra header or junk lines before the real data.
    Automatically detects which line contains the column headers.

    Parameters
    ----------
    file_bytes : bytes
        The file data from st.file_uploader().getvalue().
    max_check_lines : int
        How many lines from the start of the file to inspect.

    Returns
    -------
    df : pd.DataFrame
        Parsed DataFrame.
    header_row : int
        The detected header line index (zero-based).
    """

    # Read the first few lines to inspect
    text = file_bytes.decode(errors="ignore").splitlines()
    sample_lines = text[:max_check_lines]

    header_row = 0
    detected = False

    # Try to find the first line that looks like a header
    for i, line in enumerate(sample_lines):
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
        if len(parts) < 2:
            continue  # skip empty or short lines

        # Heuristic: if at least half the entries are non-numeric, assume this is the header
        non_numeric = 0
        for p in parts:
            try:
                float(p)
            except ValueError:
                non_numeric += 1

        if non_numeric / len(parts) >= 0.5:
            header_row = i
            detected = True
            break

    if not detected:
        header_row = 0  # fallback to first line

    # Try reading the CSV with the detected header row
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=header_row)
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), header=0)

    return df, header_row

st.set_page_config(page_title="Merge-o-matic 2000", layout="wide")
st.title("Merge-o-matic 2000 🤖")
st.write("Upload CSV or Excel files with data and timestamps. We'll time-align and filter them and let you download a combined dataset.")


# --- Step 1: File Upload ---
st.header("📁 Upload CSV, XLS, or XLSX Files")
uploaded_files = st.file_uploader(
    'Selected files',
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)


# Store data and metadata
file_data = {}
start_dates = []


if uploaded_files:
    for file in uploaded_files:
        # --- CSV files ---
        if file.name.lower().endswith(".csv"):
            # Read with flexible header detection
            raw_bytes = file.getvalue()
            df_preview, header_row = read_flexible_csv(raw_bytes, max_check_lines=30)
            df_preview = df_preview.head(2)

            # Optional: show note if data didn’t start on first line
            if header_row > 0:
                st.caption(f"Detected data starting on line {header_row + 1} in {file.name}")

        # --- Excel files ---
        else:
            raw_bytes = file.getvalue()
            df_preview = pd.read_excel(io.BytesIO(raw_bytes), nrows=2)
        file.seek(0)

        # --- Store metadata for later use ---
        file_data[file.name] = {
            "preview": df_preview,
            "columns": df_preview.columns.tolist(),
            "raw_file": raw_bytes,
            "selected_cols": {},
            "units": {},
            "cleanup": {}
        }


    st.success(f"✅ Uploaded {len(uploaded_files)} files!")


# --- Step 2: Column Selection & Cleanup ---
if file_data:
    st.header("🧹 Select Columns, Data Titles, & Cleanup Options")


    for file_name, meta in file_data.items():
        df_preview = meta["preview"]


        with st.expander(f"📈 Data available from: {file_name}", expanded=False):
            st.dataframe(df_preview)


            # --- Row 1: Column selection ---
            st.markdown(
                f"<span style='font-size:20px; font-weight:500;'>"
                f"Select data columns to include in the combination",
                unsafe_allow_html=True
            )
            selected = st.multiselect(
                "Select from the list below:",
                options=meta["columns"],
                key=f"select_{file_name}"
            )


            meta["selected_cols"] = {}
            meta["cleanup"] = {}
            meta["units"] = {}


            # --- Row 2+: Loop through each selected column ---
            for col in selected:
                with st.expander(f"⚙️ Settings for {col}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        title = st.text_input(
                            "Data Column Title",
                            value=col,
                            key=f"title_{file_name}_{col}"
                        )
                        meta["selected_cols"][col] = title
                    with col2:
                        units = st.text_input(
                            "Units",
                            value="",
                            key=f"units_{file_name}_{col}"
                        )
                    meta["units"][col] = units


                    with col3:
                        cleanup_choice = st.selectbox(
                            "Missing data handling method",
                            [
                                "Fill with nearest available value",
                                "Fill with a linear interpolation between the nearest values",
                                "Delete the entire row of data",
                                "Fill with zero"
                            ],
                            key=f"cleanup_{file_name}_{col}"
                        )
                        meta["cleanup"][col] = cleanup_choice


# --- Step 2.5: Combined Graphing Section ---
if file_data:
    st.header("📊 Visualize Combined Data (Optional)")

    if st.checkbox("Load all datasets for graphing (may take time)", value=False):
        # Gather all possible columns from all files
        available_columns = []
        for file_name, meta in file_data.items():
            for col, title in meta["selected_cols"].items():
                label = f"{file_name} - {title}"
                available_columns.append((file_name, col, title, label))

        if not available_columns:
            st.info("No selected columns available yet. Choose some in the previous step.")
        else:
            # Let user pick which ones to include on the graph
            graph_labels = [item[3] for item in available_columns]
            selected_labels = st.multiselect(
                "Select columns to display on the combined graph:",
                options=graph_labels,
                default=graph_labels[:min(3, len(graph_labels))]  # preselect a few
            )

            if selected_labels:
                combined_plot_df = pd.DataFrame()

                progress = st.progress(0, text="Loading selected data for plotting...")

                for i, (file_name, col, title, label) in enumerate(available_columns, start=1):
                    if label not in selected_labels:
                        continue

                    # --- Load file data only if needed ---
                    if file_name.lower().endswith(".csv"):
                        df_full, header_row = read_flexible_csv(file_data[file_name]["raw_file"])

                    else:
                        df_full = pd.read_excel(io.BytesIO(file_data[file_name]["raw_file"]))

                    # --- Detect datetime ---
                    datetime_cols = [c for c in df_full.columns if pd.api.types.is_datetime64_any_dtype(df_full[c]) or "date" in c.lower() or "time" in c.lower()]
                    if datetime_cols:
                        df_full[datetime_cols[0]] = pd.to_datetime(df_full[datetime_cols[0]], errors="coerce")
                        df_full = df_full.set_index(datetime_cols[0])
                    else:
                        df_full.index = pd.to_datetime(df_full.index, errors="coerce")

                    # --- Clean & sort ---
                    df_full = df_full.sort_index()
                    if df_full.index.has_duplicates:
                        df_full = df_full.groupby(df_full.index).mean(numeric_only=True)

                    y = pd.to_numeric(df_full[col], errors="coerce")
                    combined_plot_df[label] = y

                    progress.progress(i / len(available_columns), text=f"Loaded {i} of {len(available_columns)} columns")

                progress.progress(1.0, text="✅ Data ready for plotting")

                # --- Combine all into one chart ---
                combined_plot_df = combined_plot_df.sort_index()

                fig = px.line(
                    combined_plot_df,
                    x=combined_plot_df.index,
                    y=combined_plot_df.columns,
                    title="Combined Data Plot (All Files)",
                    labels={"x": "Date/Time", "value": "Value", "variable": "Column"},
                )
                fig.update_layout(
                    height=600,
                    xaxis_title="Date/Time",
                    yaxis_title="Value",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.25),
                )

                st.plotly_chart(fig, config={"responsive": True, "displaylogo": False})

# --- Step 3: Time & Interval ---
if file_data:
    st.header("🕔 Time Range & Interval")
    if start_dates:
        default_first = (max(start_dates) + pd.Timedelta(days=1))
    else:
        default_first = pd.Timestamp.today().normalize()
    default_last = default_first + pd.Timedelta(days=14)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("🗓️ Start date", value=default_first)
        range_mode = st.radio(
            "Select end range mode",
            ["End date", "Duration (days)"],
            horizontal=True
        )
    with col2:
        start_time = st.time_input("Start Time", value="00:00")
        start_dt = datetime.combine(start_date, start_time)

    if range_mode == "End date":
        col1, col2 = st.columns(2)
        with col1:
            end_date = st.date_input("End date", value=default_last)
        with col2:
            end_time = st.time_input("End Time", value="00:00")
            end_dt = datetime.combine(end_date, end_time)
    else:
        duration_days = st.number_input("Duration in days", min_value=1, value=14)
        end_dt = start_dt + pd.Timedelta(days=duration_days)
    
    interval = st.selectbox(
        "⏳ Select output time interval",
        ["1s", "30s", "1min", "5min", "10min", "15min", "1h", "1D"],
        index=2
    )

    st.subheader("📉 Interval Alignment Strategy")
    with st.expander("💡If your data is on the selected time interval, nothing will be changed. If it is not, it will be resampled according to your selections below."):
        alignment_opts = {}
        for file_name in file_data.keys():
            alignment_opts[file_name] = st.selectbox(
                f"Data from '{file_name}':",
                ["Fill with the nearest value", "Do a linear interpolation from the nearest values", "Take an average of the available values within the interval"]
            )

# --- Step 4: Create Combined File ---
if file_data:
    st.header("⛷️ Create & Download Combined File")
    if st.button("Create combined data file"):
        dt_index = pd.date_range(start=pd.Timestamp(start_dt), end=pd.Timestamp(end_dt), freq=interval)
        combined = pd.DataFrame(index=dt_index)

        row10 = ["Date"]
        row11 = [""]

        progress = st.progress(0, text="Starting file processing...")

        for i, (file_name, meta) in enumerate(file_data.items(), start=1):
            # --- Load full DataFrame from raw_file ---
            if file_name.lower().endswith(".csv"):
                df, header_row = read_flexible_csv(file_data[file_name]["raw_file"])
            else:
                # Excel: just read normally for now (we can make this flexible later)
                df = pd.read_excel(io.BytesIO(file_data[file_name]["raw_file"]))

            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower()]
            if datetime_cols:
                df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
                df = df.set_index(datetime_cols[0])
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")


            # --- Apply cleanup choices ---
            cleaned_rows = 0
            for col, method in meta["cleanup"].items():
                # Count missing values before cleanup
                missing_before = df[col].isna().sum()

                if method == "Fill with nearest available value":
                    df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
                elif method == "Fill with a linear interpolation between the nearest values":
                    df[col] = pd.to_numeric(df[col], errors="coerce").interpolate(method="time")
                elif method == "Delete the entire row of data":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df[df[col].notna()]
                elif method == "Fill with zero":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

                # If any missing values were handled, add to counter
                if missing_before > 0:
                    cleaned_rows += missing_before

            # Display cleanup info if any rows were affected
            if cleaned_rows > 0:
                st.write(f"✅ {file_name}: {cleaned_rows} missing values cleaned up.")

            if df.index.has_duplicates:
                df = df.groupby(df.index).mean(numeric_only=True)

            # --- Alignment + insert into combined ---
            for col in meta["selected_cols"]:
                if alignment_opts[file_name] == "Fill with the nearest value":
                    s = df[col].reindex(dt_index, method="nearest")
                elif alignment_opts[file_name] == "Do a linear interpolation from the nearest values":
                    s = df[col].reindex(dt_index).interpolate(method="time")
                elif alignment_opts[file_name] == "Take an average of the available values within the interval":
                    s = df[col].resample(interval).mean().reindex(dt_index)

                combined[meta["selected_cols"][col]] = s.values
                row10.append(meta["selected_cols"][col])
                row11.append(meta["units"][col])

            # --- Update progress bar ---
            progress.progress(
                i / len(file_data),
                text=f"Processed {i} of {len(file_data)} files..."
            )

        progress.progress(1.0, text="All files processed ✅")

        # Insert into Excel template
        template_path = "Analysis Template.xlsx"
        wb = openpyxl.load_workbook(template_path)
        ws = wb["Analysis"]

        # --- Grab the formatting from row 12 (template row) ---
        template_formats = {}
        for col_idx in range(2, 2 + combined.shape[1] + 1):  # B + all data cols
            template_formats[col_idx] = copy(ws.cell(row=13, column=col_idx)._style)

        # Row 10
        for c, val in enumerate(row10, start=2):
            ws.cell(row=10, column=c, value=val)

        # Row 11
        for c, val in enumerate(row11, start=2):
            ws.cell(row=11, column=c, value=val)

        # Row 12 onward
        for r, (ts, row_vals) in enumerate(combined.iterrows(), start=12):
            cell = ws.cell(row=r, column=2, value=ts)
            cell._style = copy(template_formats[2])  # Apply template style for Date/Time col

            for c, val in enumerate(row_vals, start=3):
                cell = ws.cell(row=r, column=c, value=val)
                cell._style = copy(template_formats[c])  # Apply template style for this column

        cell = ws.cell(row=12, column=2)  # Apply special formatting to first date column for charts function
        cell.number_format = "m/d/yy"

        output = BytesIO()
        wb.save(output)
        output.seek(0)
        wb.close()

        st.success("✅ Combined file created!")
        st.download_button(
            label="📥 Download Combined File",
            data=output,
            file_name="Analysis.xlsx",
        )
