import pandas as pd

from foodspace import bigideas

HEADER = ",".join(bigideas.COLUMNS)


def _write(tmp_path, name, text):
    (tmp_path / name).write_text(text)


def test_three_food_log_layouts_read_alike(tmp_path):
    _write(tmp_path, "Food_Log_001.csv", HEADER + "\n2020-02-13,18:00:00,2020-02-13 18:00:00,,Berry Smoothie,20.0,"
           "fluid ounce,Strawberry Smoothie,456.0,85.0,1.7,83.0,16.0,3.3\n")
    _write(tmp_path, "Food_Log_007.csv", "date,time,time_begin,time_end,Food_Item,Amount_Consumed,Units,"
           "Food_Item_Description,calorie,total_carb,dietary_fiber,sugar,protein,total_fat\n"
           "2020-03-14,13:44:00,2020-03-14 13:44:00,14:00:00,(Costco) Salmon,129.0,gram,Salmon,201.0,1.5,0.2,0.2,24.0,11.0\n")
    _write(tmp_path, "Food_Log_003.csv", "2020-02-22,10:30:00,2020-02-22 10:30:00,Chicken Nuggets,8.0,piece,"
           "Chicken Nuggets,393.0,19.0,0.1,20.0\n")
    _write(tmp_path, "Food_Log_015.csv", "2020-07-16,10:00:00,2020-07-16 10:00:00,,Coffee + skim milk,10,fluid ounce,,"
           "35.0,4.5,0.0,4.7,3.4,0.1\n")
    df = bigideas.read_food_logs(tmp_path)
    assert list(df["participant"]) == ["001", "003", "007", "015"]
    assert df.set_index("participant").loc["007", "logged_food"] == "(Costco) Salmon"
    n = df.set_index("participant").loc["003"]
    assert n["protein"] == 20.0 and n["sugar"] == 0.1 and pd.isna(n["total_fat"])
    assert df.set_index("participant").loc["015", "protein"] == 3.4
    assert df["time_begin"].notna().all()


def test_dexcom_keeps_glucose_readings_only(tmp_path):
    d = tmp_path / "001"
    d.mkdir()
    (d / "Dexcom_001.csv").write_text(
        "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,Patient Info,Device Info,Source Device ID,"
        "Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams),Duration (hh:mm:ss),Glucose Rate of Change (mg/dL/min),Transmitter Time (Long Integer)\n"
        "1,,FirstName,,2019,,,,,,,,\n"
        "13,2020-02-13 17:23:32,EGV,,,,iPhone G6,61.0,,,,,11101.0\n"
        "14,2020-02-13 17:28:32,EGV,,,,iPhone G6,59.0,,,,,11401.0\n")
    g = bigideas.read_dexcom(tmp_path)
    assert len(g) == 2 and g["glucose"].tolist() == [61.0, 59.0]
