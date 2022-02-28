# Exercice technique - Data scientist

Dans le cadre de notre processus de recrutement, nous aimerions vous demander de développer un classificateur ML.

# Objectif

Votre objectif est de construire un algorithme ML qui classifie automatiquement les emails par thèmes.

Pour ce faire, nous vous fournissons un jeu de données contenant :

- Un fichier texte contenant toutes les données relatives à l'email.
- Les 13 étiquettes de classification suivantes :
    1. `réglementations et régulateurs (inclut les plafonds de prix)`
    2. `projets internes -- progrès et stratégie`.
    3. Image de l'entreprise - actuelle
    4. L'image de l'entreprise - changement / influence ".
    5. influence politique / contributions / contacts ".
    6. `crise énergétique californienne / politique californienne`.
    7. `politique interne de l'entreprise`
    8. `opérations internes de l'entreprise`
    9. `alliances / partenariats`
    10. `conseils juridiques`
    11. `points de discussion`
    12. Comptes rendus de réunions
    13. `rapports de voyage`
    
- étiquettes dans la langue originale : 
    1. `regulations and regulators (includes price caps)`
    2. `internal projects -- progress and strategy`
    3. `company image -- current`
    4. `company image -- changing / influencing`
    5. `political influence / contributions / contacts`
    6. `california energy crisis / california politics`
    7. `internal company policy`
    8. `internal company operations`
    9. `alliances / partnerships`
    10. `legal advice`
    11. `talking points`
    12. `meeting minutes`
    13. `trip reports`
    

**Vous pouvez télécharger le jeu de données ici:**

[giskard_dataset.csv](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2e4249a-9c7a-44d5-9dce-a897d9d4d00e/giskard_dataset.csv)

**Exigences**

Nous voulons que vous prédisiez l'étiquette de classification en utilisant au moins les caractéristiques suivantes :

- Le contenu du courrier
- Le sujet du mail
- La date et l'heure

N'hésitez pas à ajouter toutes les caractéristiques supplémentaires auxquelles vous pourriez penser.

## Bonus : Interface W**eb**

Créez une interface web qui renvoie l'étiquette de classification prédite par votre algorithme pour des caractéristiques d'email données en entrée.
