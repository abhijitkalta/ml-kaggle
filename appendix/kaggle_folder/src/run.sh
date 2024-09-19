Now, if you want, you can create a shell script with different commands for
different folds and run them all together, as shown below.
═════════════════════════════════════════════════════════════════════════
#!/bin/sh
python train.py --fold 0
python train.py --fold 1
python train.py --fold 2
python train.py --fold 3
python train.py --fold 4
═════════════════════════════════════════════════════════════════════════
And you can run this by the following command.
═════════════════════════════════════════════════════════════════════════
❯ sh run.sh
Fold=0, Accuracy=0.8675
Fold=1, Accuracy=0.8693333333333333
Fold=2, Accuracy=0.8683333333333333
Fold=3, Accuracy=0.8704166666666666
Fold=4, Accuracy=0.8685
══════════════════════════