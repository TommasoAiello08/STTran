Vendored Action Genome annotation label files (object + predicate names only).

They match ``<AG_ROOT>/annotations/{object_classes,relationship_classes}.txt`` and are
used to initialize the Faster R-CNN detector and STTran without setting AG_DATA_PATH
(e.g. ``run_vidvrd_json_demo.py``). Full AG training / evaluation still needs AG_DATA_PATH
pointing at a complete Action Genome tree (frames + pickles).
