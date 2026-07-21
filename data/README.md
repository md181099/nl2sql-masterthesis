# Data

Die vollständigen Datensätze und generierten Datenartefakte sind nicht
Bestandteil dieses GitHub-Repositories.

Ausgeschlossen sind insbesondere:

- Spider-Rohdaten und SQLite-Datenbanken,
- SQL-Create-Context-Rohdaten,
- materialisierte Trainings-, Validierungs- und Evaluationsdateien,
- Retrievalpools,
- FAISS-Indizes und Embeddings.

Benötigte externe Datenquellen:

- Spider
- philschmid/sql-create-context-copy
- BAAI/bge-large-en-v1.5 für den Retrievalindex

Die erwarteten Datenpfade und die Schritte zur Datenaufbereitung sind in der
zentralen README des Projekts dokumentiert.

Der vollständige private Projektstand enthält unter anderem:

- 6.960 Spider-Train-Fälle,
- 18.040 ausgewählte SQL-Create-Context-Fälle,
- den kombinierten Trainingsbestand mit 25.000 Beispielen,
- MixedVal2500-v2 mit 2.500 Beispielen,
- Spider Dev mit 1.032 Evaluationsfällen,
- den Retrievalindex
  `spider_train_no_dev_overlap_bge_large_en_v15`.
