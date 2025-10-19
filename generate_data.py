"""
SmartAsset Lead Quality Dataset Generator
==========================================

Generates 2 million rows of synthetic lead data mimicking SmartAsset's 
financial advisor matching platform.

This script creates realistic lead data with:
- Demographics and financial situation
- Lead source and behavior tracking
- Engagement signals
- Advisor matching quality
- Conversion outcomes (4% baseline rate)

Run this ONCE to generate smartasset_leads.csv, then use that file for your project.

Estimated runtime: 2-5 minutes
Output file size: ~600MB
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("="*70)
print("SmartAsset Lead Data Generator")
print("="*70)
print(f"\nGenerating 2,000,000 lead records...")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
N_ROWS = 2_000_000
BASE_CONVERSION_RATE = 0.04  # 4% baseline

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_correlated_age_income_assets(n):
    """Generate age, income, and assets with realistic correlations"""
    
    # Age: normal distribution centered at 45, range 25-75
    age = np.clip(np.random.normal(45, 12, n), 25, 75).astype(int)
    
    # Income increases with age (peak earning years 45-55)
    income_base = (age - 25) * 2000 + np.random.normal(60000, 30000, n)
    income_base = np.clip(income_base, 20000, 500000)
    
    # Assets accumulate with age
    asset_base = (age - 25) * 15000 + np.random.normal(100000, 80000, n)
    asset_base = np.clip(asset_base, 0, 5000000)
    
    return age, income_base, asset_base

def assign_income_bracket(income):
    """Convert continuous income to categorical brackets"""
    if income < 50000:
        return '<50K'
    elif income < 100000:
        return '50-100K'
    elif income < 250000:
        return '100-250K'
    elif income < 500000:
        return '250-500K'
    elif income < 1000000:
        return '500K-1M'
    else:
        return '1M+'

def assign_asset_bracket(assets):
    """Convert continuous assets to categorical brackets"""
    if assets < 25000:
        return '<25K'
    elif assets < 100000:
        return '25-100K'
    elif assets < 250000:
        return '100-250K'
    elif assets < 1000000:
        return '250K-1M'
    else:
        return '1M+'

# ============================================================================
# GENERATE BASE DATA
# ============================================================================

print("Step 1/6: Generating demographics...")

# Generate correlated age, income, assets
age, income_continuous, assets_continuous = generate_correlated_age_income_assets(N_ROWS)

data = {
    'lead_id': [f'LEAD_{i:07d}' for i in range(1, N_ROWS + 1)],
    'age': age,
    'income_range': [assign_income_bracket(inc) for inc in income_continuous],
    'investable_assets': [assign_asset_bracket(ast) for ast in assets_continuous],
}

# Employment status (correlated with age)
employment_probs = np.where(age < 65, 
                           np.where(age < 60, [0.75, 0.15, 0.05, 0.05], [0.60, 0.15, 0.20, 0.05]),
                           [0.10, 0.05, 0.80, 0.05])
employment_choices = ['employed', 'self-employed', 'retired', 'other']
data['employment_status'] = np.random.choice(employment_choices, N_ROWS, p=[0.65, 0.15, 0.15, 0.05])

# Adjust employment for age (more retirees in older cohorts)
retired_mask = age > 65
data['employment_status'] = np.where(retired_mask & (np.random.random(N_ROWS) < 0.70), 
                                     'retired', data['employment_status'])

# Marital status
data['marital_status'] = np.random.choice(
    ['single', 'married', 'divorced', 'widowed'],
    N_ROWS,
    p=[0.25, 0.55, 0.15, 0.05]
)

# Children (more likely if married and age 30-55)
has_children_prob = np.where(
    (age >= 30) & (age <= 55) & (np.array(data['marital_status']) == 'married'),
    0.75, 0.30
)
data['has_children'] = (np.random.random(N_ROWS) < has_children_prob).astype(int)

# Geographic data
states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 
          'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
          'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
          'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI',
          'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
state_weights = [0.12, 0.06, 0.09, 0.065, 0.04] + [0.02] * 15 + [0.01] * 30
data['state'] = np.random.choice(states, N_ROWS, p=state_weights)

# Metro area (80% in metro)
data['metro_area'] = np.random.choice(['yes', 'no'], N_ROWS, p=[0.80, 0.20])

print("Step 2/6: Generating financial goals and situation...")

# Primary goal (correlated with age and employment)
goal_options = ['retirement', 'tax_planning', 'estate_planning', 
                'investment_advice', 'debt_management', 'home_buying']

def assign_goal(age, employment, has_children):
    if age > 55:
        return np.random.choice(['retirement', 'estate_planning', 'tax_planning'], 
                               p=[0.50, 0.30, 0.20])
    elif age < 35:
        return np.random.choice(['investment_advice', 'home_buying', 'debt_management'],
                               p=[0.40, 0.40, 0.20])
    elif employment == 'self-employed':
        return np.random.choice(['tax_planning', 'retirement', 'investment_advice'],
                               p=[0.45, 0.35, 0.20])
    elif has_children:
        return np.random.choice(['retirement', 'estate_planning', 'investment_advice'],
                               p=[0.40, 0.35, 0.25])
    else:
        return np.random.choice(goal_options)

data['primary_goal'] = [assign_goal(a, e, c) for a, e, c in 
                        zip(age, data['employment_status'], data['has_children'])]

# Retirement age target
data['retirement_age_target'] = np.where(
    np.array(data['employment_status']) == 'retired',
    age,  # Already retired
    np.clip(np.random.normal(65, 3, N_ROWS), 60, 75).astype(int)  # Target retirement age
)

# Current advisor
data['current_advisor'] = np.random.choice(['yes', 'no'], N_ROWS, p=[0.25, 0.75])

# Time horizon
data['time_horizon'] = np.random.choice(
    ['short_term', 'medium_term', 'long_term'],
    N_ROWS,
    p=[0.20, 0.40, 0.40]
)

# Risk tolerance (correlated with age - older = more conservative)
risk_probs = np.where(age < 40, [0.15, 0.45, 0.40],
                     np.where(age < 60, [0.35, 0.50, 0.15], [0.60, 0.35, 0.05]))
data['risk_tolerance'] = np.random.choice(
    ['conservative', 'moderate', 'aggressive'],
    N_ROWS,
    p=[0.40, 0.45, 0.15]  # Overall distribution
)

print("Step 3/6: Generating lead source and behavior...")

# Lead source
lead_sources = ['organic_search', 'paid_search', 'social_media', 'referral', 'direct', 'email_campaign']
data['lead_source'] = np.random.choice(
    lead_sources,
    N_ROWS,
    p=[0.35, 0.25, 0.15, 0.15, 0.07, 0.03]
)

# Device type (mobile more common for younger)
mobile_prob = np.clip(1.0 - (age - 25) / 80, 0.30, 0.70)
device_random = np.random.random(N_ROWS)
data['device_type'] = np.where(device_random < mobile_prob, 'mobile',
                               np.where(device_random < mobile_prob + 0.25, 'tablet', 'desktop'))

# Pages visited (more engaged users visit more pages)
engagement_level = np.random.beta(2, 5, N_ROWS)  # Skewed toward lower values
data['pages_visited'] = np.clip((engagement_level * 20 + np.random.normal(0, 2, N_ROWS)), 1, 30).astype(int)

# Time on site (seconds) - correlated with pages visited
data['time_on_site_seconds'] = np.clip(
    data['pages_visited'] * 45 + np.random.normal(0, 60, N_ROWS),
    30, 1800
).astype(int)

# Return visitor
data['return_visitor'] = np.random.choice([0, 1], N_ROWS, p=[0.65, 0.35])

# Form completion time (seconds)
data['form_completion_time'] = np.clip(
    np.random.lognormal(4.5, 0.8, N_ROWS),  # Log-normal distribution
    30, 900
).astype(int)

# Hour of day (normal distribution around business hours)
hour_dist = np.clip(np.random.normal(14, 4, N_ROWS), 0, 23).astype(int)
data['hour_of_day'] = hour_dist

# Day of week (slightly more on weekdays)
data['day_of_week'] = np.random.choice(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    N_ROWS,
    p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.10, 0.10]
)

print("Step 4/6: Generating engagement signals...")

# Used calculator
data['used_calculator'] = np.random.choice([0, 1], N_ROWS, p=[0.55, 0.45])

# Read article
data['read_article'] = np.random.choice([0, 1], N_ROWS, p=[0.60, 0.40])

# Calculator type (conditional on used_calculator)
calc_types = ['retirement', 'mortgage', 'investment', 'tax', 'none']
data['calculator_type'] = np.where(
    data['used_calculator'] == 1,
    np.random.choice(calc_types[:-1], N_ROWS),
    'none'
)

# Align calculator type with primary goal
goal_calc_map = {
    'retirement': 'retirement',
    'home_buying': 'mortgage',
    'investment_advice': 'investment',
    'tax_planning': 'tax'
}
for goal, calc in goal_calc_map.items():
    mask = (np.array(data['primary_goal']) == goal) & (data['used_calculator'] == 1)
    if mask.sum() > 0:
        # 70% chance to use matching calculator
        should_match = np.random.random(mask.sum()) < 0.70
        data['calculator_type'] = np.where(mask, 
                                           np.where(should_match, calc, data['calculator_type']),
                                           data['calculator_type'])

# Number of sessions
data['num_sessions'] = np.where(
    data['return_visitor'] == 1,
    np.clip(np.random.poisson(3, N_ROWS) + 2, 2, 15),
    1
)

# Email opens (only if in email campaign or return visitor)
email_eligible = (np.array(data['lead_source']) == 'email_campaign') | (data['return_visitor'] == 1)
data['email_opens'] = np.where(
    email_eligible,
    np.clip(np.random.poisson(2, N_ROWS), 0, 10),
    0
)

# Phone answered (critical conversion signal)
data['phone_answered'] = np.random.choice([0, 1], N_ROWS, p=[0.50, 0.50])

print("Step 5/6: Generating advisor match quality...")

# Advisor experience years
data['advisor_experience_years'] = np.clip(
    np.random.gamma(3, 4, N_ROWS),
    1, 40
).astype(int)

# Advisor specialization match (does advisor specialize in their goal?)
data['advisor_specialization_match'] = np.random.choice([0, 1], N_ROWS, p=[0.35, 0.65])

# Advisor rating
data['advisor_rating'] = np.clip(
    np.random.normal(4.3, 0.5, N_ROWS),
    3.0, 5.0
).round(1)

# Geographic distance
data['geographic_distance'] = np.random.choice(
    ['local', 'regional', 'national'],
    N_ROWS,
    p=[0.50, 0.35, 0.15]
)

print("Step 6/6: Calculating conversion probabilities and outcomes...")

# ============================================================================
# CALCULATE CONVERSION PROBABILITY WITH REALISTIC PATTERNS
# ============================================================================

# Start with base probability
conversion_prob = np.full(N_ROWS, BASE_CONVERSION_RATE)

# Factor 1: Investable assets (strongest predictor)
asset_multiplier = {
    '<25K': 0.3,
    '25-100K': 0.6,
    '100-250K': 1.0,
    '250K-1M': 2.0,
    '1M+': 3.5
}
for bracket, mult in asset_multiplier.items():
    mask = np.array(data['investable_assets']) == bracket
    conversion_prob[mask] *= mult

# Factor 2: Phone answered (HUGE predictor)
conversion_prob *= np.where(data['phone_answered'] == 1, 4.0, 1.0)

# Factor 3: Advisor specialization match
conversion_prob *= np.where(data['advisor_specialization_match'] == 1, 1.8, 1.0)

# Factor 4: Lead source quality
source_multiplier = {
    'referral': 1.6,
    'organic_search': 1.3,
    'direct': 1.2,
    'email_campaign': 1.0,
    'social_media': 0.9,
    'paid_search': 0.85
}
for source, mult in source_multiplier.items():
    mask = np.array(data['lead_source']) == source
    conversion_prob[mask] *= mult

# Factor 5: Engagement signals
conversion_prob *= np.where(data['used_calculator'] == 1, 1.4, 1.0)
conversion_prob *= np.where(data['return_visitor'] == 1, 1.3, 1.0)

# Factor 6: Time on site (more time = more serious)
time_boost = np.clip(data['time_on_site_seconds'] / 600, 0.7, 1.5)
conversion_prob *= time_boost

# Factor 7: Device type (desktop slightly better)
device_multiplier = {'desktop': 1.2, 'tablet': 1.0, 'mobile': 0.9}
for device, mult in device_multiplier.items():
    mask = np.array(data['device_type']) == device
    conversion_prob[mask] *= mult

# Factor 8: Age sweet spot (55-65 is peak)
age_factor = np.where(
    (age >= 55) & (age <= 65), 1.4,
    np.where((age >= 45) & (age < 55), 1.2,
            np.where(age < 35, 0.7, 1.0))
)
conversion_prob *= age_factor

# Factor 9: Already has advisor (lower conversion)
conversion_prob *= np.where(np.array(data['current_advisor']) == 'yes', 0.6, 1.0)

# Factor 10: Late night form fills (2am-4am) = lower quality
late_night = (hour_dist >= 2) & (hour_dist <= 4)
conversion_prob *= np.where(late_night, 0.5, 1.0)

# Factor 11: Very fast form completion (suspicious)
very_fast = data['form_completion_time'] < 45
conversion_prob *= np.where(very_fast, 0.6, 1.0)

# Factor 12: Advisor rating boost
rating_boost = (data['advisor_rating'] - 3.5) * 0.3 + 1
conversion_prob *= rating_boost

# Factor 13: Local advisor preference
conversion_prob *= np.where(np.array(data['geographic_distance']) == 'local', 1.2, 1.0)

# Factor 14: Thursday boost (arbitrary but realistic quirk)
conversion_prob *= np.where(np.array(data['day_of_week']) == 'Thursday', 1.15, 1.0)

# Clip probabilities to [0, 1]
conversion_prob = np.clip(conversion_prob, 0, 1)

# Generate conversions based on probabilities
data['converted'] = (np.random.random(N_ROWS) < conversion_prob).astype(int)

# ============================================================================
# INTRODUCE MODERATE MESSINESS
# ============================================================================

print("\nIntroducing realistic data quality issues...")

# 1. Missing values (2-3% in select columns)
missing_columns = ['income_range', 'advisor_rating', 'time_on_site_seconds', 
                   'email_opens', 'advisor_experience_years']

for col in missing_columns:
    if col in data:
        missing_mask = np.random.random(N_ROWS) < 0.025  # 2.5% missing
        if col == 'income_range' or col == 'investable_assets':
            data[col] = np.where(missing_mask, None, data[col])
        else:
            data[col] = np.where(missing_mask, np.nan, data[col])

# 2. One obvious data quality issue: some negative ages (data entry error)
error_mask = np.random.random(N_ROWS) < 0.001  # 0.1% bad data
data['age'] = np.where(error_mask, -data['age'], data['age'])

# 3. Add timestamp for realism
start_date = datetime(2024, 1, 1)
days_range = 365
random_days = np.random.randint(0, days_range, N_ROWS)
random_seconds = np.random.randint(0, 86400, N_ROWS)
timestamps = [start_date + timedelta(days=int(d), seconds=int(s)) 
              for d, s in zip(random_days, random_seconds)]
data['submission_date'] = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]

# ============================================================================
# CREATE DATAFRAME AND SAVE
# ============================================================================

print("\nCreating DataFrame...")
df = pd.DataFrame(data)

# Reorder columns logically
column_order = [
    'lead_id',
    'submission_date',
    'converted',
    # Demographics
    'age',
    'income_range',
    'investable_assets',
    'employment_status',
    'marital_status',
    'has_children',
    'state',
    'metro_area',
    # Financial goals
    'primary_goal',
    'retirement_age_target',
    'current_advisor',
    'time_horizon',
    'risk_tolerance',
    # Lead source & behavior
    'lead_source',
    'device_type',
    'pages_visited',
    'time_on_site_seconds',
    'return_visitor',
    'form_completion_time',
    'hour_of_day',
    'day_of_week',
    # Engagement
    'used_calculator',
    'read_article',
    'calculator_type',
    'num_sessions',
    'email_opens',
    'phone_answered',
    # Advisor match
    'advisor_experience_years',
    'advisor_specialization_match',
    'advisor_rating',
    'geographic_distance'
]

df = df[column_order]

# Save to CSV
output_file = 'smartasset_leads.csv'
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("DATASET GENERATED SUCCESSFULLY!")
print("="*70)

file_size_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\nDataset Statistics:")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Memory usage: {file_size_mb:.1f} MB")
print(f"  File saved: {output_file}")

print(f"\nConversion Statistics:")
print(f"  Total conversions: {df['converted'].sum():,}")
print(f"  Conversion rate: {df['converted'].mean():.2%}")
print(f"  (Target was ~{BASE_CONVERSION_RATE:.1%})")

print(f"\nData Quality:")
print(f"  Missing values by column:")
for col in df.columns:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"    {col}: {missing:,} ({missing/len(df):.2%})")

print(f"\n  Data quality issues to fix:")
print(f"    Negative ages: {(df['age'] < 0).sum():,} rows")

print(f"\nTop Predictor Preview:")
assets_conversion = df.groupby('investable_assets')['converted'].mean().sort_values(ascending=False)
print(f"  Conversion by Assets:")
for asset, rate in assets_conversion.items():
    print(f"    {asset}: {rate:.2%}")

print("\n" + "="*70)
print("Ready to use! Load with: pd.read_csv('smartasset_leads.csv')")
print("="*70)
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
