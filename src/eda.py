import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
os.makedirs('eda', exist_ok=True)

# Load CSVs
results = pd.read_csv('data/ufc_fight_results.csv')
events = pd.read_csv('data/ufc_event_details.csv')
stats = pd.read_csv('data/ufc_fight_stats.csv')
fighters = pd.read_csv('data/ufc_fighter_tott.csv')

# Outcome distribution
results['OUTCOME'].value_counts().plot(kind='bar', figsize=(8, 6))
plt.title('Fight Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda/outcome_distribution.png')
plt.close()

# Weight class distribution (top 15)
weightclass_counts = results['WEIGHTCLASS'].value_counts().head(15)
weightclass_counts.plot(kind='barh', figsize=(10, 8))
plt.title('Top 15 Weight Classes')
plt.xlabel('Number of Fights')
plt.tight_layout()
plt.savefig('eda/weightclass_distribution.png')
plt.close()

# Method distribution
results['METHOD'].value_counts().head(10).plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Fight End Methods')
plt.xlabel('Method')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda/method_distribution.png')
plt.close()

# Round distribution
results['ROUND'].value_counts().sort_index().plot(kind='bar', figsize=(8, 6))
plt.title('Fight End Round Distribution')
plt.xlabel('Round')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda/round_distribution.png')
plt.close()

# Fights over time
events['DATE'] = pd.to_datetime(events['DATE'], errors='coerce')
events_by_year = events.groupby(events['DATE'].dt.year).size()
events_by_year.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Number of Events Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.grid(True)
plt.tight_layout()
plt.savefig('eda/events_over_time.png')
plt.close()

# Top locations
events['LOCATION'].value_counts().head(15).plot(kind='barh', figsize=(10, 8))
plt.title('Top 15 Fight Locations')
plt.xlabel('Number of Events')
plt.tight_layout()
plt.savefig('eda/location_distribution.png')
plt.close()

# Fighter stance distribution
fighters['STANCE'].value_counts().plot(kind='bar', figsize=(8, 6))
plt.title('Fighter Stance Distribution')
plt.xlabel('Stance')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('eda/stance_distribution.png')
plt.close()

# Fighter height distribution (parse height first)
def parse_height(value):
    if pd.isna(value) or value == '--':
        return None
    try:
        parts = str(value).replace('"', '').split("'")
        if len(parts) == 2:
            return int(parts[0].strip()) * 12 + int(parts[1].strip())
    except:
        return None
    return None

fighters['height_inches'] = fighters['HEIGHT'].apply(parse_height)
fighters['height_inches'].dropna().hist(bins=30, figsize=(10, 6))
plt.title('Fighter Height Distribution')
plt.xlabel('Height (inches)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda/height_distribution.png')
plt.close()

# Fighter weight distribution
def parse_weight(value):
    if pd.isna(value) or value == '--':
        return None
    try:
        return int(str(value).replace(' lbs.', ''))
    except:
        return None

fighters['weight_lbs'] = fighters['WEIGHT'].apply(parse_weight)
fighters['weight_lbs'].dropna().hist(bins=30, figsize=(10, 6))
plt.title('Fighter Weight Distribution')
plt.xlabel('Weight (lbs)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda/weight_distribution.png')
plt.close()

# Fighter reach distribution
def parse_reach(value):
    if pd.isna(value) or value == '--':
        return None
    try:
        return int(str(value).replace('"', ''))
    except:
        return None

fighters['reach_inches'] = fighters['REACH'].apply(parse_reach)
fighters['reach_inches'].dropna().hist(bins=30, figsize=(10, 6))
plt.title('Fighter Reach Distribution')
plt.xlabel('Reach (inches)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda/reach_distribution.png')
plt.close()

# Missing values in fighter attributes
missing = fighters[['HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB']].apply(
    lambda col: (col == '--').sum() + col.isnull().sum()
)
missing.plot(kind='bar', figsize=(10, 6))
plt.title('Missing Values in Fighter Attributes')
plt.xlabel('Attribute')
plt.ylabel('Missing Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda/fighter_missing_values.png')
plt.close()

# Stats distribution - parse and aggregate
def parse_fraction_landed(value):
    if pd.isna(value) or value in ['---', '--']:
        return 0
    try:
        parts = str(value).split(' of ')
        return int(parts[0]) if len(parts) == 2 else 0
    except:
        return 0

stats['sig_strikes_landed'] = stats['SIG.STR.'].apply(parse_fraction_landed)
stats_agg = stats.groupby(['EVENT', 'BOUT', 'FIGHTER'])['sig_strikes_landed'].sum().reset_index()

stats_agg['sig_strikes_landed'].hist(bins=50, figsize=(10, 6))
plt.title('Significant Strikes Landed per Fighter per Fight')
plt.xlabel('Strikes Landed')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda/strikes_distribution.png')
plt.close()

print("EDA charts saved to eda/ folder")

