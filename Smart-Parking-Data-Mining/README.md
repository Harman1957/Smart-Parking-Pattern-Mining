# Smart Parking Pattern Mining and Demand Analysis Using Data Mining Techniques

## Project purpose
This BCA-level project analyzes historical parking records to discover patterns in demand, duration, vehicle type, and slot usage. It is a **data mining and analysis project**, not a parking booking/management application.

## Important dataset note
`dataset/parking_data.csv` is a **synthetic sample dataset created for practice**. It is not real-world parking data. Clearly identify it as synthetic in your report and presentation.

## Tools
- Python
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- mlxtend
- Power BI Desktop
- VS Code (Google Colab is an alternative)

## Quick start in VS Code (Windows)
1. Extract this ZIP.
2. Open the extracted `Smart-Parking-Data-Mining` folder in VS Code.
3. Open **Terminal > New Terminal**.
4. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, tell your instructor/assistant the exact message; you can also select `.venv` as the Python interpreter in VS Code.
6. Install libraries:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Run the starter pipeline:
   ```powershell
   python src\run_analysis.py
   ```
8. Open `results/` to view generated CSVs, charts, and mining outputs.

If `python` is not recognized, try `py` in the commands.

## Workflow
1. Data inspection, cleaning, and preprocessing
2. Exploratory data analysis
3. K-Means clustering
4. Apriori association rule mining
5. Decision Tree classification
6. Outlier detection
7. Power BI dashboard

## Dataset fields
- `Record_ID`: row/session identifier
- `Date`, `Entry_Time`, `Exit_Time`: date/time details
- `Vehicle_Type`, `Vehicle_ID`, `Slot_ID`: vehicle and slot details
- `Parking_Duration_Min`: sample parking duration in minutes
- `Parking_Fee`: example fee value
- `Day`, `Day_Type`: day information

## Important interpretation notes
- The sample data and its results are for practice only, not actual observations from a real parking facility.
- The starter classification task creates demand labels from hourly counts to demonstrate a Decision Tree. It is not a validated real-world forecasting model.
- The slot summary counts sessions per slot; true occupancy/utilization rates require time-window and capacity calculations.
- The `.pbix` dashboard must be created and saved in Power BI Desktop. This ZIP contains dashboard instructions, not a finished Power BI report.
