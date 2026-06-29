import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc

from configs.paths import PREFERENCE_FILE, FIGURE_DIR, METADATA_DIR
from configs.preferences import BASE_PROFILES
from configs.dataset import RANDOM_SEED, NUM_SYNTHETIC_PREFERENCES

PREFERENCE_FILE.parent.mkdir(exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def generate_user_preferences(n_preferences=NUM_SYNTHETIC_PREFERENCES, seed=RANDOM_SEED):
    np.random.seed(seed)

    anchors = np.array(list(BASE_PROFILES.values()))

    # 1. Generate LHS untuk coverage ruang [0,1]³ yang merata & tanpa lubang
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    lhs_samples = sampler.random(n=n_preferences)

    # 2. Ganti ~30% sampel dengan preferensi realistis berbasis anchor + noise
    n_anchors = int(0.3 * n_preferences)
    replace_idx = np.random.choice(n_preferences, n_anchors, replace=False)

    chosen_anchors = anchors[np.random.randint(0, len(anchors), size=n_anchors)]
    noise = np.random.normal(0, 0.1, (n_anchors, 3))
    anchor_samples = np.clip(chosen_anchors + noise, 0.0, 1.0)

    # Gabungkan & acak agar tidak bias urutan
    final_preferences = lhs_samples.copy()
    final_preferences[replace_idx] = anchor_samples
    np.random.shuffle(final_preferences)

    return final_preferences

# GENERATE PREFERENCES
print("Generating synthetic preferences (LHS + Anchor)...")
user_preferences = generate_user_preferences(n_preferences=NUM_SYNTHETIC_PREFERENCES, seed=RANDOM_SEED)

print(f"Generated: {len(user_preferences)}")
print(f"Shape: {user_preferences.shape}")

user_preferences = user_preferences.astype(np.float32)

pref_df = pd.DataFrame(
    user_preferences,
    columns=[
        "bass",
        "mid",
        "treble"
    ]
)

pref_df.insert(
    0,
    "preference_id",
    range(len(pref_df))
)

pref_df.to_csv(METADATA_DIR / "preferences_metadata.csv", index=False)
np.save(PREFERENCE_FILE, user_preferences)

print(f"\nSaved to: {PREFERENCE_FILE}")

# --- VISUALIZATION 1: Histogram per dimens ---
fig = plt.figure(figsize=(12,4))

plt.subplot(131)
plt.hist(user_preferences[:,0], bins=15, edgecolor='black', alpha=0.7)
plt.title("Bass Preference")
plt.xlabel("Preference")
plt.ylabel("Count")

plt.subplot(132)
plt.hist(user_preferences[:,1], bins=15, edgecolor='black', alpha=0.7)
plt.title("Mid Preference")
plt.xlabel("Preference")

plt.subplot(133)
plt.hist(user_preferences[:,2], bins=15, edgecolor='black', alpha=0.7)
plt.title("Treble Preference")
plt.xlabel("Preference")
plt.tight_layout()

save_name_1 = "Histogram_per_Dimension.png"
save_fig_1 = FIGURE_DIR / "dataset" / save_name_1
plt.savefig(save_fig_1, format='png', dpi=300)
plt.close()

# --- VISUALIZATION 2: 3D Scatter ---
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(user_preferences[:,0], user_preferences[:,1], user_preferences[:,2],
                c=user_preferences[:,0], cmap='viridis', s=60, edgecolor='k', alpha=0.8)
ax.set_xlabel('Bass'); ax.set_ylabel('Mid'); ax.set_zlabel('Treble')
ax.set_title(f'Distribusi {len(user_preferences)} Preferensi (LHS + Anchor)')
plt.colorbar(sc, label='Bass Level')

save_name_2 = "Preference_3D_Scatter.png"
save_fig_2 = FIGURE_DIR / "dataset" / save_name_2
plt.savefig(save_fig_2, format='png', dpi=300)
plt.close()