to use tel_churn.csv(one hot encoded data) we need to use ANN in model.ipynb and use compile method as 'categorical_crossentropy' and activation='softmax'(on last layer)

or to use first_telc.csv only on ANN use compile method as 'sparse_categorical_crossentropy' and activation='softmax'(on last layer)


we must choose sparse method
