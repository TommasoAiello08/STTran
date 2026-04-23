# APT su Colab — QUICKSTART (A100, Pro+)

Aggiornato: 23 aprile 2026

Obiettivo: allenare e valutare APT (Dynamic Scene Graph Generation via
Anticipatory Pre-training, CVPR 2022) su Action Genome partendo da zero su
una sessione Colab con A100 (la GPU più potente disponibile su Colab).

---

## TL;DR in 30 secondi

1. Su **Google Drive** metti questi tre asset una volta sola (vedi §1).
2. Su **Colab** apri `scripts/colab_train_apt.ipynb` (3 celle).
3. Esegui in ordine le 3 celle. L'ultima stampa un
   `training_report.txt` che puoi incollarmi per continuare il lavoro.

Non devi caricare a mano il codice del repo: la cella 1 fa `git clone` dal
tuo fork GitHub `https://github.com/TommasoAiello08/STTran.git`.

---

## 1) Cosa ti serve su Google Drive (una volta sola)

```
/content/drive/MyDrive/
├── action_genome/                 ← il dataset Action Genome
│   ├── annotations/
│   │   ├── object_classes.txt
│   │   ├── relationship_classes.txt
│   │   ├── object_bbox_and_relationship.pkl
│   │   └── person_bbox.pkl
│   └── frames/
│       └── <video_id>/*.png       ← ~300 GB se completo; ok anche subset
├── apt_assets/
│   └── faster_rcnn_ag.pth         ← checkpoint Faster R-CNN (~530 MB)
│                                     link Drive nel README di yrcong/STTran
└── apt_ckpts/                     ← creata dal runner per checkpoint + report
```

Note:
- Il dataset Action Genome si ottiene dagli autori originari (Charades +
  annotation pack). Se hai solo i video MP4 e non i frame estratti, serve
  un passo di `ffmpeg` preliminare (non trattato qui).
- `faster_rcnn_ag.pth` è scaricabile dal Google Drive di yrcong/STTran
  (link nel README originale). Copialo in `apt_assets/` su Drive.
- Lo spazio per i checkpoint APT è ~200 MB × 2 stage × `nepoch` epoche.
  Drive Pro+ (200 GB) basta.

---

## 2) Cosa caricare su Colab

**Niente a mano.** Le 3 celle del notebook fanno tutto:

- **Cella 1** monta Google Drive ed esegue `git clone` del repo (dal tuo
  fork). Copia automaticamente `faster_rcnn_ag.pth` da
  `apt_assets/` dentro `fasterRCNN/models/`.
- **Cella 2** lancia `scripts/colab_run_all.py --stage all`, l'unico file
  da eseguire. Questo orchestratore:
  1. compila l'estensione CUDA di Faster R-CNN (`model._C`);
  2. compila le due estensioni Cython (`draw_rectangles`, `bbox`);
  3. scarica GloVe 6B in `data/`;
  4. (opzionale) copia il dataset da Drive a `/content` per I/O veloce;
  5. esegue gli smoke test CPU per validare build + modello;
  6. Stage 1 pre-training (SGD lr=1e-3, batch 8, 10 epoche, AMP on);
  7. Stage 2 fine-tuning (SGD lr=1e-5, batch 8, 10 epoche, AMP on);
  8. valuta PredCls / SGCls / SGGen con 3 vincoli;
  9. scrive `training_report.txt` su `apt_ckpts/` in Drive.
- **Cella 3** stampa il `training_report.txt` sì che tu possa incollarmelo.

---

## 3) Come eseguire (passo-passo)

1. Vai su [Colab](https://colab.research.google.com).
2. File → Open notebook → GitHub → incolla `TommasoAiello08/STTran`
   (o carica via File → Open notebook → Upload se preferisci), seleziona
   `scripts/colab_train_apt.ipynb`.
3. Runtime → Change runtime type → **GPU (A100)** → Save.
4. Esegui **Cella 1**. Autorizza l'accesso a Drive. Deve stampare il
   commit SHA e `Copied Faster R-CNN checkpoint into fasterRCNN/models/.`
5. Esegui **Cella 2**. Questo è il run principale. Dura:
   - ~20 h su A100 fp16 per la ricetta completa 10+10 epoche predcls;
   - meno se imposti `--pretrain_overrides nepoch=3 --finetune_overrides
     nepoch=3` nella cella.
   I checkpoint sono salvati ogni epoca su `apt_ckpts/` in Drive; se
   la sessione cade, ri-esegui la cella — il flag `--resume` riparte dal
   `_latest.tar`.
6. Esegui **Cella 3** e incolla in chat il contenuto di
   `training_report.txt`.

---

## 4) Varianti utili

### Solo sanity test (senza training vero, ~5 min)

Modifica Cella 2 così:
```
!python scripts/colab_run_all.py --stage smoke --ag_root $AG_ROOT --ckpt_root $CKPT_ROOT
```
Verifica solo build estensioni + smoke test CPU del modello.

### Run "mini" per verificare che il training giri davvero sulla GPU (~1 h)

Usa il config mini:
```
!python scripts/colab_run_all.py --stage all \
    --ag_root $AG_ROOT --ckpt_root $CKPT_ROOT --mode predcls \
    --pretrain_config configs/apt_pretrain_colab_mini.yaml \
    --pretrain_overrides nepoch=1 \
    --finetune_overrides nepoch=1
```

### Copy dataset to local SSD for fast I/O

A100 su Colab Pro+ ha ~200 GB di disco locale. Se i tuoi `frames/`
occupano meno di 150 GB, aggiungi `--copy_to_local` alla Cella 2:

```
!python scripts/colab_run_all.py --stage all --copy_to_local ...
```

Il runner fa rsync da Drive a `/content/action_genome/` una sola volta
(poi il copy è skipped se già presente). Aspettarsi 10× sulla velocità
di lettura rispetto a Drive montato.

### Solo SGDet (detector non-GT)

Cambia `--mode predcls` in `--mode sgdet`. Raccomandato batch più
piccolo e lambda ridotto:
```
--pretrain_overrides batch_size=4 lambda=6 \
--finetune_overrides batch_size=4 lambda=6
```

---

## 5) Cosa finisce nel `training_report.txt`

- Ambiente: versione Python, torch, nome GPU (dovrebbe dire A100), VRAM.
- Comando completo invocato.
- Statistiche dataset: `data_path` usato, numero di cartelle video,
  eventuali path mancanti.
- Stage 1: config usato, tail degli ultimi log epoch (loss a/s/c + lr),
  path dei checkpoint salvati.
- Stage 2: idem.
- Evaluation: R@20 / R@50 / R@100 per `with_constraint`, `no_constraint`,
  `semi_constraint`.
- Wall-clock totale.
- Un JSON gemello con gli stessi dati strutturati.

Incollami il TXT a fine run e posso fare ispezioni mirate su loss,
metriche, eventuali errori, e pianificare lo step successivo.

---

## 6) Checklist pre-esecuzione

- [ ] Runtime Colab è impostato su GPU A100 (non CPU).
- [ ] Drive contiene `action_genome/annotations/` e `action_genome/frames/`.
- [ ] Drive contiene `apt_assets/faster_rcnn_ag.pth`.
- [ ] Il fork GitHub a cui punta Cella 1 è aggiornato (push del tuo lavoro
      locale prima di aprire Colab).
- [ ] Hai memoria Drive sufficiente (~1-2 GB per i checkpoint APT).

Se una condizione manca, la Cella 2 si ferma e il
`training_report.txt` include comunque la diagnostica che mi permette di
aiutarti a risolvere.
