import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def comptar_i_representar_classes(y, title="Distribució de Classes"):
    """
    Compta les instàncies de cada classe i les representa amb un gràfic de barres,
    afegint el nombre exacte d'instàncies a sobre de cada barra.
    
    Args:
    - y: Conjunt de dades amb les etiquetes (sèries o array).
    - title: Títol del gràfic.
    """
    # Comptar instàncies per classe
    class_counts = pd.Series(y).value_counts().sort_index()

    # Mostrar els comptatges per terminal
    print("Comptatge d'instàncies per classe:")
    print(class_counts)

    # Crear gràfic de barres
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'instàncies")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Afegir valors exactes a sobre de cada barra
    for i, value in enumerate(class_counts.values):
        ax.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=10, color='black')

    plt.show()

# Exemple d'ús amb el conjunt d'entrenament abans del balanceig
train_data = pd.read_csv("data/train_set.csv")
y_train = train_data['score']
comptar_i_representar_classes(y_train, title="Distribució de Classes")
