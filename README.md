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
