import pandas as pd

def format_hour(hour):
    return pd.to_datetime(str(hour), format="%H").strftime("%I %p")

def get_appliance_insights(data):
    df = pd.DataFrame(data)

    # Extract hour of day for start and end
    df["start_hour"] = df["start_time"].dt.hour
    df["end_hour"] = df["end_time"].dt.hour

    # Most frequent start and end hour per appliance
    most_common_times = df.groupby("appliance_name").agg(
        most_common_start=("start_hour", lambda x: x.mode()[0]),
        most_common_end=("end_hour", lambda x: x.mode()[0])
    ).reset_index()

    # Format into human-readable string
    output_strings = []
    for _, row in most_common_times.iterrows():
        appliance = row["appliance_name"].replace("_", " ")
        start = format_hour(row["most_common_start"])
        end = format_hour(row["most_common_end"])
        output_strings.append(
            f"The {appliance} most commonly starts at {start} and finishes at {end}."
        )

    # Join into a single report
    report = "\n".join(output_strings)

    return report
