# Power BI dashboard instructions

After running the Python pipeline, open Power BI Desktop and import:
- `results/cleaned_parking_data.csv`
- `results/sessions_by_hour.csv`
- `results/sessions_by_day.csv`
- `results/slot_usage.csv`
- CSV files in `results/mining_results/`

Suggested pages:
1. Parking Overview: sessions, average duration, fee total, vehicle type.
2. Parking Demand: sessions by hour/day and demand categories.
3. Slot Usage: sessions per slot and average duration per slot.
4. Mining Results: clusters, association rules, classification report, potential outliers.

Save the report here as `Smart_Parking_Dashboard.pbix`. A finished `.pbix` is not included because it needs to be authored in Power BI Desktop.
