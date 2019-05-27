# AlphaNPI

Adapting the alphaGo algorithm  to remove the need of execution traces to train NPI.

## Setup
You need to install the required Python packages.

    cd alphanpi/

Then run the command:

    pip install -r requirements.txt


Update the following environment variable:

    export PYTHONPATH=$PWD:$PYTHONPATH

## Training

    cd trainings/

Run one of the scripts:

    python train_recursive_sorting.py --tensorboard --verbose --save-model --save-results --save-model
    python train_hanoi.py --tensorboard --verbose --save-model --save-results --save-model
    python train_sorting_nohierarchy.py --tensorboard --verbose --save-model --save-results --save-model
    python train_hanoi.py --tensorboard --verbose --save-model --save-results --save-model
For more information about the arguments that can be sent, see at trainings/README.md
    

## Validation

The following allows to assert the results disclosed in the paper

cd validation/ then run one of the scripts:


    python validate_hanoi.py --verbose --save-results
    python validate_recursive_sorting.py --verbose --save-results
    python validate_sorting.py --verbose --save-results
    python validate_sorting_nohierarchy.py --verbose --save-results
For more information about the arguments that can be sent, see at validation/README.md


### Results disclosed in the paper:
#### Sorting and Recursive sorting

| Length | Non-recursive MCTS | Non-recursive Network only | Recursive MCTS | Recursive Network only |
| ------ | ------ | ------ | ------ | ------ |
| 10 | 100% | 85% | 100% | 70% |
| 20 | 100% | 85% | 100% | 60% |
| 60 | 95% | 40% | 100% | 35% |
| 100 | 40% | 10% | 100% | 10% |

#### Sorting with no hierarchy

| Length | Non-recursive MCTS | Non-recursive Network only |
| ------ | ------ | ------ |
| 3 | 94% | 78% |
| 4 | 42% | 22% |
| 5 | 10% | 5% |
| 6 | 1% | 1% |

#### Tower of Hanoi

| Number of disks | MCTS | Network only |
| ------ | ------ | ------ |
| 2 | 100% | 100% |
| 5 | 100% | 100% |
| 10 | 100% | 100% |
| 12 | 100% | 100% |
    
## Visualization

    cd visualization/

To visualize a pre-trained model **behavior** for an environment **env**, run the script **visualize_{env}.py**. Set the load path at the beginning of the script to the path where the model of interest is saved.

Run one of the scripts:

    python visualize_hanoi.py
    python visualize_recursive_sorting.py
    python visualize_sorting.py
    python visualize_sorting_nohierarchy.py
    
### Generate visualization
When a visualization script is executed, it generates a **mcts.gv** file under visualization/. The file contains a description of the tree in dot language. 

If you don't already have graphviz installed, run command:

    sudo apt-get install python3-pydot graphviz

To convert the .gv in .pdf file, use command:

    dot -Tpdf mcts.gv -o mcts.pdf



    
    