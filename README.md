Reinforcement Learning -- Cost-Aware Warehouse Navigation
========================================================

Dieses Projekt implementiert eine kostenbasierte Pfadoptimierung in einem Warehouse-ähnlichen Umfeld mittels Reinforcement Learning. Ziel ist es zu zeigen, wie ein Agent mithilfe von Q-Learning nicht nur den kürzesten, sondern den kosteneffizientesten Pfad zu einem Ziel erlernen kann.\
Das Projekt basiert konzeptionell auf einem praxisorientierten Fachartikel zur Warehouse-Optimierung mit Q-Learning und wurde eigenständig erweitert, insbesondere durch gestufte Bottlenecks und eine interaktive Visualisierungsanwendung.

* * * * *

Projektüberblick
----------------

Das Projekt besteht aus drei zentralen Komponenten:

**Gridworld-Environment**\
Modelliert ein Warehouse als diskretes Gitter mit freien Zellen, blockierten Wänden und kostenintensiven Engpässen (Bottlenecks).

**Q-Learning-Agent**\
Lernt durch Interaktion mit dem Environment eine Policy zur kostenbasierten Pfadoptimierung.

**Interaktive Visualisierungs-App**\
Ermöglicht das Setzen von Start und Ziel, das Training des Agenten sowie die visuelle Analyse der Ergebnisse.

* * * * *

Projektstruktur
---------------

ReinforcementLearning\
├── gridworld\
│ ├── maps.py -- Grid- und Map-Generator (Walls, Bottlenecks, Zufallskarten)\
│ └── env.py -- GridWorld-Environment (gym-ähnlich)\
├── q_learning\
│ └── agent.py -- Q-Learning-Agent (Q-Tabelle, Update-Regel)\
├── visualization\
│ └── app.py -- Interaktive Visualisierungs- und Steuerungs-App\
├── README.md -- Projektdokumentation\
└── requirements.txt -- Abhängigkeiten

* * * * *

Voraussetzungen
---------------

Python 3.10 oder neuer\
Empfohlen wird die Verwendung eines virtuellen Environments.

### Benötigte Python-Bibliotheken

-   numpy

-   matplotlib

Installation der Abhängigkeiten:

pip install -r requirements.txt

* * * * *

Konzeptionelles Modell
----------------------

### Gridworld / Warehouse

Jede Zelle entspricht einem Zustand.\
Der Agent kann sich in vier Richtungen bewegen: oben, unten, links und rechts.\
Walls sind nicht begehbar.\
Bottlenecks sind begehbar, verursachen jedoch zusätzliche Kosten.

### Bottleneck-Level

Bottlenecks sind gestuft modelliert, um reale Engpässe abzubilden:

Level 1: Leichter Engpass (z. B. leichter Stau)\
Level 2: Mittlerer Engpass\
Level 3: Schwerer Engpass (z. B. Unfall)

Je höher das Level, desto höher die Zusatzkosten.

* * * * *

Reward-Design (Kostenmodell)
----------------------------

Das Reward-Design definiert, welches Verhalten der Agent als vorteilhaft oder nachteilig erlernt.\
Ziel ist es, nicht primär die Weglänge, sondern die minimierten Gesamtkosten eines Pfads zu optimieren.

| Ereignis | Beschreibung | Reward |
| --- | --- | --- |
| Normaler Schritt | Bewegung auf eine freie Zelle | -1 |
| Ziel erreicht | Erreichen des Zielzustands | +100 |
| Ungültiger Schritt | Bewegung gegen Wall oder ausserhalb der Karte | -10 |
| Bottleneck Level 1 | Leichter Engpass (z. B. leichter Stau) | -6 |
| Bottleneck Level 2 | Mittlerer Engpass | -12 |
| Bottleneck Level 3 | Schwerer Engpass (z. B. Unfall) | -18 |

Durch diese Kostenstruktur lernt der Agent, Trade-offs zwischen Weglänge und Zusatzkosten zu berücksichtigen.\
Ein kürzerer Pfad mit hohen Engpasskosten kann somit schlechter bewertet sein als ein längerer, aber kostengünstigerer Umweg.

* * * * *

Start der Anwendung (Visualisierung)
------------------------------------

### App starten

Wechsle ins Projektverzeichnis und starte die Visualisierungs-App mit:

**python -m visualization.app**

Nach dem Start öffnet sich ein interaktives Fenster.

* * * * *

Bedienung der App
-----------------

### Karte (Gridworld)

Linke Seite: Warehouse-Grid\
Schwarze Felder: Walls (blockiert)\
Schraffierte Felder mit Zahlen: Bottlenecks (Level 1--3)

### Start und Ziel setzen

Start und Ziel können entweder per Mausklick oder per Texteingabe gesetzt werden.

Per Klick auf die Karte erfolgt die Auswahl über die Radio-Buttons:\
Set Start (click)\
Set Goal (click)

Alternativ ist eine Texteingabe im Format row,col möglich (z. B. 0,0).

### Training starten

Über den Button „Train + Show" wird der Trainingsprozess gestartet.\
Der Agent trainiert über mehrere Episoden.\
Nach Abschluss werden automatisch visualisiert:

Greedy-Pfad\
Return-Kurve\
Value-Heatmap\
Policy-Pfeile

### Map wechseln

Der Button „Change Map" erzeugt eine neue zufällige Karte mit garantierter Erreichbarkeit sowie neuer Anordnung von Walls und Bottlenecks.

### Ansicht zurücksetzen

Der Button „Reset View" entfernt die Trainingsergebnisse, behält jedoch die aktuelle Karte bei.

* * * * *

Ergebnisdarstellung
-------------------

### Greedy-Pfad

Zeigt den vom Agenten gewählten kostenoptimalen Pfad.\
Dieser ist nicht zwingend der kürzeste Pfad.

### Return-Kurve

Stellt den Gesamtreward pro Episode dar.\
Sie dient zur Analyse des Lernfortschritts und der Konvergenz.

### Value-Heatmap

Visualisiert den maximalen Q-Wert pro Zustand.\
Hohe Werte deuten auf attraktive Zustände hin.

### Policy-Pfeile

Zeigen die bevorzugte Aktion pro Zustand.\
Sie ermöglichen eine lokale Interpretation der erlernten Policy.

* * * * *

Wissenschaftlicher Kontext
--------------------------

Dieses Projekt wurde im Modul Reinforcement Learning umgesetzt.\
Als konzeptionelle Grundlage diente ein praxisorientierter Fachartikel:

Foy, P. (2019).\
AI for ecommerce: Optimizing business processes with reinforcement learning.\
MLQ.ai\
[https://blog.mlq.ai/ai-for-ecommerce-optimizing-business-processes/](https://blog.mlq.ai/ai-for-ecommerce-optimizing-business-processes/?utm_source=chatgpt.com)

Der Fachartikel wurde konzeptionell genutzt, insbesondere für die Modellierung eines Warehouse als diskretes Environment, den Einsatz von Q-Learning zur Pfadoptimierung sowie die grundlegende Idee der Kostenabbildung über Bottleneck-Penalties.\
Die gesamte Implementierung, das Reward-Design, die Erweiterungen sowie die Visualisierung wurden vollständig eigenständig entwickelt.

* * * * *

Erweiterungsmöglichkeiten
-------------------------

Zeitabhängige Bottlenecks\
Mehrere Agenten\
Reale Karten als Graphen (z. B. Stadtmodelle)\
Deep Q-Networks (DQN) für grössere Zustandsräume\
Integration von Echtzeitdaten (z. B. Verkehr)

* * * * *

Autor
-----

Armor Ajdini\
Bachelor of Science in Business Artificial Intelligence\
Fachhochschule Nordwestschweiz FHNW