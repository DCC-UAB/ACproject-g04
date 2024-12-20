import pandas as pd

# --- Dataset en anglès ---
# Carregar el dataset anglès
data_path_english = "data/TripAdvisor_reviews.csv"
df_english = pd.read_csv(data_path_english)

# Filtrar només les files amb valoracions de 4 o 5
filtered_df_english = df_english[df_english['rating_review'].isin([4, 5])]

# Comptar instàncies per a les classes 4 i 5
class_counts_english = filtered_df_english['rating_review'].value_counts()
print("Dataset anglès:")
print(f"Nombre d'instàncies abans del filtratge: {len(df_english)}")
print(f"Nombre d'instàncies després del filtratge: {len(filtered_df_english)}")
print(f"Nombre d'instàncies amb valoració 4: {class_counts_english.get(4, 0)}")
print(f"Nombre d'instàncies amb valoració 5: {class_counts_english.get(5, 0)}")

# Guardar el dataset filtrat en un nou fitxer CSV
output_path_english = "data/4_5_english.csv"
filtered_df_english.to_csv(output_path_english, index=False)
print(f"Dataset filtrat anglès guardat a: {output_path_english}")

# --- Dataset en castellà ---
# Carregar el dataset castellà
data_path_spanish = "data/ressenyes_es.csv"
df_spanish = pd.read_csv(data_path_spanish)

# Filtrar només les files amb valoracions de 4 o 5
filtered_df_spanish = df_spanish[df_spanish['rating'].isin([4, 5])]

# Comptar instàncies per a les classes 4 i 5
class_counts_spanish = filtered_df_spanish['rating'].value_counts()
print("\nDataset castellà:")
print(f"Nombre d'instàncies abans del filtratge: {len(df_spanish)}")
print(f"Nombre d'instàncies després del filtratge: {len(filtered_df_spanish)}")
print(f"Nombre d'instàncies amb valoració 4: {class_counts_spanish.get(4, 0)}")
print(f"Nombre d'instàncies amb valoració 5: {class_counts_spanish.get(5, 0)}")

# Guardar el dataset filtrat en un nou fitxer CSV
output_path_spanish = "data/4_5_castella.csv"
filtered_df_spanish.to_csv(output_path_spanish, index=False)
print(f"Dataset filtrat castellà guardat a: {output_path_spanish}")
