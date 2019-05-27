# Visualization

To visualize a pre-trained model **behavior** for an environment **env**, run the script **visualize_{env}.py**. Set the load path at the beginning of the script to the path where the model of interest is saved.

## Generate visualization
When a visualization script is executed, it generates a **mcts.gv** file under visualization/. The file contains a description of the tree in dot language. 

If you don't already have graphviz installed, run command:

    sudo apt-get install python3-pydot graphviz

To convert the .gv in .pdf file, use command:

    dot -Tpdf mcts.gv -o mcts.pdf
