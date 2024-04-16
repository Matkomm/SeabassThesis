import pandas as pd

# Load the Excel file
#file_path = r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Litteratur\Fra Nikos\Parametere\fish_data2.xlsx'
fish_data = pd.read_excel(file_path, sheet_name='fish_data2')
environment_data = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert the timestamp columns to datetime, letting pandas infer the format
fish_data['observed At:'] = pd.to_datetime(fish_data['observed At:'], errors='coerce')
environment_data['formatted_timestamp'] = pd.to_datetime(environment_data['formatted_timestamp'], errors='coerce')

print(fish_data['observed At:'].dtype)
print(environment_data['formatted_timestamp'].dtype)
#If the conversion is successful, the dtype should now be 'datetime64[ns]'

# You can also check for rows where the datetime conversion failed
fish_data[fish_data['observed At:'].isnull()]
environment_data[environment_data['formatted_timestamp'].isnull()]

# Convert timezone-aware datetime objects to timezone-naive
fish_data['observed At:'] = fish_data['observed At:'].dt.tz_localize(None)
environment_data['formatted_timestamp'] = environment_data['formatted_timestamp'].dt.tz_localize(None)

environment_data = environment_data.sort_values('formatted_timestamp')

# Initialize columns to hold the matched temperature and oxygen data
fish_data['temperature'] = pd.NA
fish_data['dissolved Oxygen'] = pd.NA

# Iterate over each fish observation to find the nearest prior environmental record
for i, fish_row in fish_data.iterrows():
    # Filter to get all prior environmental records
    mask = environment_data['formatted_timestamp'] <= fish_row['observed At:']
    if not environment_data.loc[mask].empty:
        # Get the closest prior record
        closest_record = environment_data.loc[mask].iloc[-1]

        # Update the fish data with environmental info
        fish_data.at[i, 'temperature'] = closest_record['Water Temperature']
        fish_data.at[i, 'dissolved Oxygen'] = closest_record['Dissolved Oxygen MG']

with pd.ExcelWriter(r'c:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Litteratur\Fra Nikos\Parametere\fish_data2_test.xlsx', 
                    engine='xlsxwriter',
                    datetime_format='yyyy-mm-dd hh:mm:ss.00') as writer:  # Customize the format as needed
    fish_data.to_excel(writer, index=False)

#fish_data.to_excel(r'c:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Litteratur\Fra Nikos\Parametere\fish_data2_merge.xlsx', index=False)
