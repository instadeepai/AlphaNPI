# Validation

All experiment files accept several command line parameters. Each experiment file run validation on a pre-trained model for a specific environment.

## Seed
All experiments can be seeded. To set a seed for an experiment, use argument:

	--seed seed_number
	
## Set number of cpus 
The number of cpus used for each experiment can be set (if not precsied, the default number is 8) with the argument:

    --num-cpus number_of_cpus
    
## Pre-trained model path
Specify the pre-trained model path with:

    --load-path path_to_model

## Print in console
To print training progress in console, use argument:

	--verbose

## Save results in .txt file
Validation results may be saved in a .txt file in the folder **Results**. To do it, use argument:

	--save-results