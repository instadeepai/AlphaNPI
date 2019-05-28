# AlphaNPI

Adapting the AlphaZero algorithm  to remove the need of execution traces to train NPI.

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

    cd validation/

Then run one of the scripts:


    python validate_hanoi.py --verbose --save-results
    python validate_recursive_sorting.py --verbose --save-results
    python validate_sorting.py --verbose --save-results
    python validate_sorting_nohierarchy.py --verbose --save-results
For more information about the arguments that can be sent, see at validation/README.md

    
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



    
    
