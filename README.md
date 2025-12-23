## Reinforcement Learning -- Cost-Aware Warehouse Navigation

Dieses Projekt implementiert eine kostenbasierte Pfadoptimierung in einem Warehouse-ähnlichen Umfeld mittels Reinforcement Learning. Ziel ist es zu zeigen, wie ein Agent mithilfe von Q-Learning nicht nur den kürzesten, sondern den **kosteneffizientesten Pfad** zu einem Ziel erlernen kann.\
Das Projekt basiert konzeptionell auf einer bestehenden Studie zur Warehouse-Optimierung mit Q-Learning und wurde eigenständig erweitert, insbesondere durch gestufte Bottlenecks und eine interaktive Visualisierungsanwendung.

* * * * *

**Projektüberblick**

Das Projekt besteht aus drei zentralen Komponenten:

1.  **Gridworld-Environment**\
    Modelliert ein Warehouse als diskretes Gitter mit freien Zellen, blockierten Wänden und kostenintensiven Engpässen (Bottlenecks).
2.  **Q-Learning-Agent**\
    Lernt durch Interaktion mit dem Environment eine Policy zur kostenbasierten Pfadoptimierung.
3.  **Interaktive Visualisierungs-App**\
    Ermöglicht das Setzen von Start und Ziel, das Training des Agenten sowie die visuelle Analyse der Ergebnisse.

* * * * *

**Projektstruktur**

ReinforcementLearning/

│

├── gridworld/

│   ├── maps.py          # Grid- und Map-Generator (Walls, Bottlenecks, Zufallskarten)

│   └── env.py           # GridWorld-Environment (gym-ähnlich)

│

├── q_learning/

│   └── agent.py         # Q-Learning-Agent (Q-Tabelle, Update-Regel)

│

├── visualization/

│   └── app.py           # Interaktive Visualisierungs- und Steuerungs-App

│

├── README.md            # Dieses Dokument

└── requirements.txt     # Abhängigkeiten

* * * * *

**Voraussetzungen**

-   Python **3.10 oder neuer**
-   Empfohlen: virtuelles Environment

**Benötigte Python-Bibliotheken**

-   numpy
-   matplotlib

Installation der Abhängigkeiten:

pip install -r requirements.txt

* * * * *

**Konzeptionelles Modell**

**Gridworld / Warehouse**

-   Jede Zelle entspricht einem **Zustand**
-   Bewegungen: oben, unten, links, rechts
-   **Walls** sind nicht begehbar
-   **Bottlenecks** sind begehbar, verursachen jedoch zusätzliche Kosten

**Bottleneck-Level**

Bottlenecks sind gestuft modelliert, um reale Engpässe abzubilden:

-   **Level 1**: leichter Engpass (z. B. leichter Stau)
-   **Level 2**: mittlerer Engpass
-   **Level 3**: schwerer Engpass (z. B. Unfall)

Je höher das Level, desto höher die Zusatzkosten.

* * * * *

**Reward-Design (Kostenmodell)**

Das Reward-Design definiert, welches Verhalten der Agent als vorteilhaft oder nachteilig erlernt. Ziel ist es, nicht den kürzesten, sondern den **kosteneffizientesten Pfad** zu optimieren.

| Ereignis               | Beschreibung                                      | Reward |
|------------------------|--------------------------------------------------|--------|
| Normaler Schritt       | Bewegung auf eine freie Zelle                    | -1     |
| Ziel erreicht          | Erreichen des Zielzustands                       | +100   |
| Ungültiger Schritt     | Bewegung gegen Wall oder ausserhalb der Karte    | -10    |
| Bottleneck Level 1     | Leichter Engpass (z. B. leichter Stau)           | -6     |
| Bottleneck Level 2     | Mittlerer Engpass                                | -12    |
| Bottleneck Level 3     | Schwerer Engpass (z. B. Unfall)                  | -18    |

Durch diese Kostenstruktur lernt der Agent, **Trade-offs zwischen Weglänge und Zusatzkosten** zu berücksichtigen.  
Ein kürzerer Pfad mit hohen Engpasskosten kann somit schlechter bewertet sein als ein längerer, aber kostengünstigerer Umweg.

* * * * *

**Start der Anwendung (Visualisierung)**

**App starten**

Wechsle ins Projektverzeichnis und starte die Visualisierungs-App:

**python visualization/app.py**

Nach dem Start öffnet sich ein interaktives Fenster.

* * * * *

**Bedienung der App**

**1\. Karte (Gridworld)**

-   Linke Seite: Warehouse-Grid
-   Schwarze Felder: Walls (blockiert)
-   Schraffierte Felder mit Zahlen: Bottlenecks (Level 1--3)

**2\. Start und Ziel setzen**

-   **Per Klick auf die Karte**\
    Mit den Radio-Buttons unten links auswählen:

-   „Set Start (click)"
-   „Set Goal (click)"

-   **Oder per Texteingabe**

-   Format: row,col (z. B. 0,0)

**3\. Training starten**

-   Button **„Train + Show"**
-   Der Agent trainiert über mehrere Episoden
-   Danach werden automatisch visualisiert:

-   Greedy-Pfad
-   Return-Kurve
-   Value-Heatmap
-   Policy-Pfeile

**4\. Map wechseln**

-   Button **„Change Map"**
-   Erzeugt eine neue zufällige Karte mit:

-   garantierter Erreichbarkeit
-   neuen Walls und Bottlenecks

**5\. Ansicht zurücksetzen**

-   Button **„Reset View"**
-   Entfernt Trainingsergebnisse, behält die Karte bei

* * * * *

**Ergebnisdarstellung**

**Greedy-Pfad**

-   Zeigt den vom Agenten gewählten kostenoptimalen Pfad
-   Nicht zwingend der kürzeste Pfad

**Return-Kurve**

-   Zeigt den Gesamtreward pro Episode
-   Dient zur Analyse von Lernfortschritt und Konvergenz

**Value-Heatmap**

-   Visualisiert den maximalen Q-Wert pro Zustand
-   Hohe Werte deuten auf attraktive Zustände hin

**Policy-Pfeile**

-   Zeigen die bevorzugte Aktion pro Zustand
-   Erlauben eine lokale Interpretation der Policy

* * * * *

**Wissenschaftlicher Kontext**

Dieses Projekt wurde im Modul **Reinforcement Learning** umgesetzt.\
Die zugrunde liegende Studie:

Foy, P. (2019). *AI for ecommerce: Optimizing business processes with reinforcement learning*. MLQ.ai.

wurde **konzeptionell** genutzt, insbesondere für:

-   Warehouse als diskretes Environment
-   Q-Learning-Ansatz
-   Grundidee von Bottleneck-Penalties

Der gesamte Code, das Reward-Design, die Erweiterungen sowie die Visualisierung wurden **eigenständig entwickelt**.

* * * * *

**Erweiterungsmöglichkeiten**

-   Zeitabhängige Bottlenecks
-   Mehrere Agenten
-   Reale Karten als Graphen (z. B. Stadtmodelle)
-   Deep Q-Networks (DQN) für grössere Zustandsräume
-   Echtzeitdaten (z. B. Verkehr)

* * * * *

**Autor**

Armor Ajdini\
Bachelor of Science in Business Artificial Intelligence\
Fachhochschule Nordwestschweiz FHNW# ReinforcementLearning
