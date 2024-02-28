import pandas as pd
from datetime import datetime, timedelta

def fill_missing_minutes(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Convert the date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create a list to hold the new rows
    new_rows = []
    
    # Iterate through the original DataFrame to fill missing minutes
    for index, row in df.iterrows():
        new_rows.append({'Date': row['Date'], 'Value': row['Value']})

        if index < len(df) - 1:
            diff = (df.loc[index + 1, 'Date'] - row['Date']).seconds // 60
            if diff > 1:
                for i in range(1, diff):
                    new_row = {'Date': row['Date'] + timedelta(minutes=i), 'Value': None}
                    new_rows.append(new_row)
    
    # Create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)
    
    # Save the filled DataFrame to a new CSV file
    new_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_filename = "cip.csv"  # Replace with your input CSV filename
    output_filename = "out_cip.csv"  # Replace with the desired output CSV filename
    fill_missing_minutes(input_filename, output_filename)
    print("Missing minutes filled and saved successfully.")



