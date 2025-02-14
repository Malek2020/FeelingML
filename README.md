# 📌 API d'Analyse des Sentiments - Guide d'Installation et d'Utilisation

## 1️⃣ Étapes d'Installation
Suivez ces étapes pour configurer et exécuter l'API.

### ✅ Étape 1 : Cloner le dépôt
```bash
git clone <https://github.com/Malek2020/FeelingML.git>
cd <FeelingML_API>
```

### ✅ Étape 2 : Installer les dépendances
```bash
pip install -r requirements.txt
```

### ✅ Étape 3 : Créer la base de données MySQL nommée `sentiment_db`
```sql
CREATE DATABASE sentiment_db;

USE sentiment_db;
CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    positive TINYINT(1) NOT NULL,
    negative TINYINT(1) NOT NULL
);
```

### ✅ Étape 4 : Insérer des données pour l'entraînement
```sql
INSERT INTO tweets (text, positive, negative) VALUES
("J'adore ce produit, il est parfait !", 1, 0),
("C'est la pire expérience de ma vie, une honte !", 0, 1);
```

### ✅ Étape 5 : Exécuter le script
```bash
python FeelingML.py
```
