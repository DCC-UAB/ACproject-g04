import matplotlib.pyplot as plt

def grafica_repartiment_dades(train_size, val_size, test_size):
    """
    Crea un gràfic de sectors per mostrar el repartiment de les dades en Train, Validation i Test.
    
    Args:
    - train_size: Proporció de dades per al conjunt d'entrenament.
    - val_size: Proporció de dades per al conjunt de validació.
    - test_size: Proporció de dades per al conjunt de test.
    """
    # Etiquetes per a cada secció
    labels = ['Train (70%)', 'Validation (15%)', 'Test (15%)']
    
    # Mida de cada secció
    sizes = [train_size, val_size, test_size]
    
    # Colors per a cada secció
    colors = ['#66b3ff', '#99ff99', '#ff6666']
    
    # Crear gràfic de sectors
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
    
    # Títol del gràfic
    plt.title('Repartiment de les Dades: Train, Validation i Test')
    plt.show()

# Cridar la funció amb els percentatges de repartiment
grafica_repartiment_dades(70, 15, 15)
