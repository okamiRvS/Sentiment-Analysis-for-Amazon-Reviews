## Sentiment Analysis for Amazon-Reviews
* Umberto Cocca - 807191

# Introduzione
Negli ulitmi anni si è visto un numero crescente di ricerche che hanno ampliato la comprensione del sentiment delle risorse testuali determinando l’avvento di servizi online che hanno cambiato il volto allo shopping.

Applicazioni di commercio online come Amazon concepiscono una quantità spropositata di dati per mezzo delle transizioni e degli utenti di questo servizio, infatti una parte consistente è data dai contenuti generati dagli utenti che valutano i prodotti acquistati e
condividono la loro esperienza procedendo con valutazioni numeriche, seguite spesso da delle recensioni.

# Sentiment Analysis
Il Sentiment Analysis serve per interpretare il linguaggio naturale e identificare informazioni soggettive che denotano opinioni, emozioni e sentimenti, determinndo la polarità corrispondente (positiva, negativa o neutra) e comprendere il soggetto / oggetto target.
In questa fase, viene analizzata sistematicamente le parti testuali delle recensioni per estrarne un’opinione. Una parte preliminare pre-processing servirà per preparare il dataset. Vengono scartate recensioni troppo lunghe o troppe corte.
Infine, viene utilizzato ASUM (Aspet Sentiment Unification Model) per poter estrarre quelle che sono un insieme di topic che sono riferiti ai sentiment positivi e negativi. Usando ASUM si assume che il documento sia composto da frasi.

# Software utilizzati
Il flusso di lavoro si compone di due aree, una fase di manipolazione dei dati attraverso Python e una fase di processione dei dati attraverso il modello ASUM per il sentiment analysis.

*Python*\
Per eseguire il preprocessing si è lavorato con Python, scelta dovuta alla grande quantità di strumenti e librerie open source disponibili per questo linguaggio. Le librerie utilizzate sono le seguenti:
* *Pandas*: per caricare e manipolare il dataset
* *NLTK*: per separare ogni review in una lista di frasi
* *re*: per eseguire una pulizia parziale sui dati, ad esempio eliminando quelle parole composte solo da numeri, o da caratteri inadeguati.

*ASUM*\
Per mezzo di Python si è costruito l’input ad hoc per la versione Java di ASUM creata da Yohan Jo and Alice Oh, consultabile al seguente link.
L’input del programma è costituito da tre file, due obbligatori e uno opzionale:
* *BagOfSentences.txt (obbligatorio)*\
Questo file è una rappresentazione dell’elenco di parole dei documenti nel corpus. Per ogni documento, la prima riga è il numero di frasi. Dalla riga successiva viene visualizzato un elenco di indici che si riferiscono alla posizione relativa nel WordList;
* *WordList.txt (obbligatorio)*\
Questo file mappa le parole con indici di parole. Ogni parola è scritta in una riga. Si presume che la prima parola nel file abbia l’indice 0, la seconda parola abbia indice 1 e così via...;
* SentiWords-0.txt, SentiWords-1.txt, . . . (opzionale)*\
Questi file sono parole chiamati "semi sentimentali". Il numero del file dovrebbe iniziare da 0 e aumentare gradualmente. Nel modello ASUM è possibile aiutare il processo di campionamento facendo uso di queste informazioni a priori. Se sappiamo che una determinata parola è positiva perch´e appartiene al lessico dei positivi allora la sua probabilità di essere positiva la si conosce. Nello specifico per il progetto si sono usati due sentiment, uno positivo e uno negativo sfruttando due liste di parole italiane recuperate dal seguente link e manipolato leggermente aggiungendo alcune emoticon testuali.
