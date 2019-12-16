# ga-hyperparam-tuning
Find optimal hyper-parameters for Neural Networks using Genetic Algorithms

This is a "just for fun" DIY project to find optimal set of hyperparameters for a Neural Network. For now it only supports MLPs (plain old Neural Networks) but if it works on practical projects and if I get good feedback, I will ConvNets and LSTMs too.

# How to Use this?
Since I don't have enough feedback on it's performance, I have not uploaded a installable package on pypi yet. So to use this project:

1. you'll have to copy the files in 'src' folder to your project
2. create a search space of hyperparameters which will act as a boundry for search (more on this later)
3. define the required functions like evaluation function which would try out your model and return loss/accuracy.
4. decide an optimization mode; 'max' if you want to maximize the result (e.g. while using accuracy) or 'min' while using loss.
5. then finally call GAEngine like this:
```
GAEngine(search_space_mlp, mutation_probability=0.4, exit_check=exit_check,
                                 on_generation_end=on_generation_end, func_eval=func_eval,
                                 population_size=5, opt_mode=mode).ga_search()
```                                 

The functions "exit_check" and "on_generation_end" give you better control over when to end the search. All these examples can be found in the main.py file included in the project.
I will be writing a detailed blog on this and post the link here.

# How to define search space?
Defining search space of parameters is very simple and intuitive. If there is some parameter for which you want to choose one of the values from give choices, pass it as a python list ("[]"). Example learning rate as 'lr': [0.1, 0.01]. On the other if there is some parameter for which you want to select either the whole group of parameters or nothing, pass it as dict ("{}"). Example would be "layers" of MLP for which either you would select one layer or two layers etc, each with different set of values for number of nodes, dropout. Also don't forget to include the unchanged parameters like input nodes, output nodes. The defaul function which uses these params to create Keras model uses this info. If you want to use your own function to create model then this can be skipped.

# Sample:
```
search_space_mlp = {
    'input_size': 200,
    'batch_size': [40, 60, 80, 100, 120, 150],
    'layers': [
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid']
        },
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': list(np.arange(10, 501)),
            'do_layer_2': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_2': ['relu', 'sigmoid']
        },
        {
            'nodes_layer_1': list(np.arange(10, 501)),
            'do_layer_1': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_1': ['relu', 'sigmoid'],

            'nodes_layer_2': list(np.arange(10, 501)),
            'do_layer_2': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_2': ['relu', 'sigmoid'],

            'nodes_layer_3': list(np.arange(10, 501)),
            'do_layer_3': list(np.linspace(0, 0.5, dtype=np.float32)),
            'activation_layer_3': ['relu', 'sigmoid']
        }
    ],
    "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    "epochs": [3000],
    "optimizer": ["rmsprop", "sgd", "adam"],
    'output_nodes': 3,
    'output_activation': 'softmax',
    'loss': 'categorical_crossentropy'
}
```
