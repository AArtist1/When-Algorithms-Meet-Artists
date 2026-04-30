"""
Frame Analysis Code
Copy and paste any or all of these cells into Create_Artist_Perspectives.ipynb
Assumes df_survey is already loaded from:
    df_survey = pd.read_csv("data/lovato_artist/ai_art_surveydata_cleaned.csv")
"""

# ============================================================
# CELL 1: Setup and filtering
# ============================================================

import pandas as pd
import numpy as np

# If df_survey is not already loaded, uncomment:
# df_survey = pd.read_csv("data/lovato_artist/ai_art_surveydata_cleaned.csv")

# Filter to US-based, self-identified artists (matching manuscript: n=252)
df_artists = df_survey[
    (df_survey['Artist'].str.strip() == 'Yes') &
    (df_survey['Country'].str.contains('United States', na=False))
].copy()

print(f"US-based artists: {len(df_artists)}")

# Map Likert values (1=Disagree, 2=Neutral, 3=Agree)
LIKERT = {1.0: 'Disagree', 2.0: 'Neutral', 3.0: 'Agree'}

df_artists['threat'] = df_artists['AI_threat_2art_workers'].map(LIKERT).fillna('No response')
df_artists['utility'] = df_artists['AI_pos_development_4art'].map(LIKERT).fillna('No response')
df_artists['transparency'] = df_artists['Required_disclosure'].map(LIKERT).fillna('No response')
df_artists['own_user'] = df_artists['Styleof_ownedby_AI_user'].map(LIKERT).fillna('No response')
df_artists['own_artist'] = df_artists['Styleof_ownedby_OG_artist'].map(LIKERT).fillna('No response')
df_artists['own_company'] = df_artists['Styleof_ownedby_AI_company'].map(LIKERT).fillna('No response')
df_artists['comp'] = df_artists['compensation'].fillna('No response').astype(str).str.strip()


# ============================================================
# CELL 2: Threat x Utility cross-tab
# ============================================================

print("THREAT x UTILITY CROSS-TAB")
print("=" * 50)
ct = pd.crosstab(df_artists['threat'], df_artists['utility'], margins=True)
print(ct)

print("\nAs percentages:")
ct_pct = pd.crosstab(df_artists['threat'], df_artists['utility'], normalize='all') * 100
print(ct_pct.round(1))

# The key stat
both = len(df_artists[(df_artists['threat'] == 'Agree') & (df_artists['utility'] == 'Agree')])
print(f"\nAgree BOTH threat AND utility: {both} ({100*both/len(df_artists):.1f}%)")


# ============================================================
# CELL 3: Threat-Utility combined labels
# ============================================================

def threat_utility_label(row):
    t, u = row['threat'], row['utility']
    if t == 'Agree' and u == 'Agree':     return 'Dual: Threat + Utility'
    if t == 'Agree' and u == 'Disagree':   return 'Threat-Dominant'
    if t == 'Agree' and u == 'Neutral':    return 'Threat-Leaning'
    if t == 'Disagree' and u == 'Agree':   return 'Utility-Dominant'
    if t == 'Disagree' and u == 'Disagree': return 'Dual Skeptic'
    if t == 'Disagree' and u == 'Neutral': return 'Low-Threat Uncertain'
    if t == 'Neutral' and u == 'Agree':    return 'Utility-Leaning'
    if t == 'Neutral' and u == 'Disagree': return 'Cautious Skeptic'
    if t == 'Neutral' and u == 'Neutral':  return 'Undecided'
    return 'Other'

df_artists['threat_utility'] = df_artists.apply(threat_utility_label, axis=1)

print("THREAT x UTILITY COMBINED LABELS:")
print(df_artists['threat_utility'].value_counts().to_string())


# ============================================================
# CELL 4: Compensation labels
# ============================================================

COMP_LABELS = {
    'donate_to_trainer': 'Commons Donor',
    'nocomp_but_no_forprofit': 'Anti-Corporate',
    'flat_fee': 'Flat Fee',
    'portion_from_model_creators': 'Revenue Share (Creators)',
    'portion_from_derivatives': 'Revenue Share (Derivatives)',
    'portion_of_any_profit_made': 'Revenue Share (Any Profit)',
    'not_comfortable_with_any_listed_options': 'Rejects All Options',
    'I_dont_need_profit': 'No Profit Needed',
    'nocompt_noprofit4anyone': 'Anti-Profit',
    'tax': 'AI Tax',
    'Other': 'Other',
    'No response': 'No response',
}

df_artists['comp_label'] = df_artists['comp'].map(COMP_LABELS).fillna(df_artists['comp'])

print("COMPENSATION STANCES:")
print(df_artists['comp_label'].value_counts().to_string())


# ============================================================
# CELL 5: Full frame combinations
# ============================================================

# Build a combined frame string using 5 main dimensions
df_artists['frame_combo'] = (
    df_artists['threat'] + ' | ' +
    df_artists['utility'] + ' | ' +
    df_artists['transparency'] + ' | ' +
    df_artists['own_artist'] + ' | ' +
    df_artists['comp']
)

n_unique = df_artists['frame_combo'].nunique()
print(f"Unique frame combinations (5 dimensions): {n_unique}")

print(f"\nTOP 20 COMBINATIONS:")
for i, (combo, count) in enumerate(df_artists['frame_combo'].value_counts().head(20).items(), 1):
    print(f"  {i:2d}. [{count:3d}, {100*count/len(df_artists):4.1f}%] {combo}")


# ============================================================
# CELL 6: Dimension distributions (for supplementary table)
# ============================================================

print("\nDIMENSION DISTRIBUTIONS:")
print("=" * 50)

for dim_name, col in [('Threat', 'threat'), ('Utility', 'utility'),
                       ('Transparency', 'transparency'), ('Ownership (artist)', 'own_artist')]:
    print(f"\n{dim_name}:")
    for val, count in df_artists[col].value_counts().items():
        print(f"  {val:15s}: {count:3d} ({100*count/len(df_artists):5.1f}%)")

print(f"\nCompensation:")
for val, count in df_artists['comp_label'].value_counts().items():
    print(f"  {val:30s}: {count:3d} ({100*count/len(df_artists):5.1f}%)")


# ============================================================
# CELL 7: Sample profiles for Introduction table
# ============================================================

print("\nSAMPLE ARTIST PROFILES:")
print("=" * 50)

profiles = [
    {
        'label': 'The Pragmatic Dual-Holder',
        'filter': (df_artists['threat'] == 'Agree') & (df_artists['utility'] == 'Agree') &
                  (df_artists['transparency'] == 'Agree') & (df_artists['own_artist'] == 'Agree') &
                  (df_artists['comp'] == 'donate_to_trainer'),
    },
    {
        'label': 'The Protective Advocate',
        'filter': (df_artists['threat'] == 'Agree') & (df_artists['utility'] == 'Disagree') &
                  (df_artists['transparency'] == 'Agree') & (df_artists['own_artist'] == 'Agree') &
                  (df_artists['comp'] == 'portion_from_model_creators'),
    },
    {
        'label': 'The Open-Source Embracer',
        'filter': (df_artists['threat'] == 'Disagree') & (df_artists['utility'] == 'Agree') &
                  (df_artists['own_artist'] == 'Disagree') &
                  (df_artists['comp'] == 'donate_to_trainer'),
    },
    {
        'label': 'The Cautious Observer',
        'filter': (df_artists['threat'] == 'Neutral') & (df_artists['utility'] == 'Neutral') &
                  (df_artists['transparency'] == 'Agree'),
    },
]

for p in profiles:
    matches = df_artists[p['filter']]
    if len(matches) > 0:
        row = matches.iloc[0]
        print(f"\n  {p['label']} (n={len(matches)}):")
        print(f"    Threat:       {row['threat']}")
        print(f"    Utility:      {row['utility']}")
        print(f"    Transparency: {row['transparency']}")
        print(f"    Ownership:    {row['own_artist']}")
        print(f"    Compensation: {COMP_LABELS.get(row['comp'], row['comp'])}")
    else:
        print(f"\n  {p['label']}: No exact match")


# ============================================================
# CELL 8: Ownership cross-tab (all 3 sub-questions)
# ============================================================

print("\nOWNERSHIP: Who should own AI art in the artist's style?")
print("=" * 50)
for owner, col in [('Original Artist', 'own_artist'), ('AI User', 'own_user'), ('AI Company', 'own_company')]:
    a = (df_artists[col] == 'Agree').sum()
    d = (df_artists[col] == 'Disagree').sum()
    n = (df_artists[col] == 'Neutral').sum()
    print(f"  {owner:20s}: Agree={a:3d} ({100*a/len(df_artists):.1f}%), "
          f"Disagree={d:3d} ({100*d/len(df_artists):.1f}%), "
          f"Neutral={n:3d} ({100*n/len(df_artists):.1f}%)")


# ============================================================
# CELL 9: The "34 frames" vs "137 combinations" question
# ============================================================

# The manuscript says "34 distinct frames." Where does 34 come from?
# It's the number of unique perspective_text values in the filtered dataset.
df_persp = pd.read_csv("data/artist_perspectives.csv")
# Filter same way as manuscript: question_group != 'familiarity'
df_persp_filtered = df_persp[df_persp['question_group'] != 'familiarity']

n_unique_texts = df_persp_filtered['perspective_text'].nunique()
n_unique_per_group = df_persp_filtered.groupby('question_group')['perspective_text'].nunique()

print(f"\nTHE '34 FRAMES' QUESTION:")
print(f"  Unique perspective_text values (excl familiarity): {n_unique_texts}")
print(f"\n  By question group:")
for group, count in n_unique_per_group.items():
    print(f"    {group:15s}: {count}")

print(f"\n  The 137 number = unique CROSS-DIMENSIONAL combinations")
print(f"  The {n_unique_texts} number = unique WITHIN-DIMENSION response texts")
print(f"  Both are correct; they count different things.")
