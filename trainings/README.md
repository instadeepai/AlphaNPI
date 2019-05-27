# Trainings

All experiment files accept several command line parameters. Each experiment file runs a training for a specific environment.

## All trainings
### Seed
All experiments can be seeded. To set a seed for an experiment, use argument:

	--seed seed_number
	
### Set number of cpus 
The number of cpus used for each experiment can be set (if not precised, the default number is 8) with the argument:

    --num-cpus number_of_cpus
    
### Tensorboard

To display the training progress in Tensorboard, use argument:

	--tensorboard
	
Then to launch tensorboard, run the command

	tensorboard --logdir runs
	
### Save model

To save the model at every iteration inside the folder **models**, use argument:

	--save-model

### Print in console
To print training progress in console, use argument:

	--verbose

### Save results in .txt file
Training progress may also be saved in a .txt file in the folder **results**. To do it, use argument:

	--save-results


## Sorting no hierarchy

### Set maximum training length 
The max length of the lists that will be used for training (if not precised, the default number is 3) with the argument:

    --max-length max_length
    
### Set validation length 
The length of the lists that will be used to validate the results (if not precised, the default number is 3) with the argument:

    --val-length val_length