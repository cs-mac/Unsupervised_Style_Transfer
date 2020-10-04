# Unsupervised Style Transfer: Automatic Sentiment Transfer Using Classification Attention Weights

Code for my Master's Thesis Information Science at the University of Groningen.

### How to run the models
First preprocess the data using: ``preprocess.py``

Before we can train the HAN model we first need to get the POS-tags using: ``HAN/POS.ipynb``

Next we can train the HAN model using: ``HAN/HAN+POS_attention_mechanism.ipynb``

We can then use the ``style_generator.ipynb`` to generate sentences from one sentiment to the opposing sentiment

We can use ``train_evaluation.ipynb`` to train the classifiers for automatic classification

Lastly, we can use ``evaluation.ipynb`` and ``all_evaluation.ipynb`` to evaluate the human and automatic evaluations

```diff
Note:
- This is research code and might therefore not be fully complete. 
- For questions and full results contact the author.
```

## Authors

* **Chi Sam Mac** - [Github](https://github.com/cs-mac/) - [E-Mail](chisam_mac@hotmail.com) - [LinkedIn](https://www.linkedin.com/in/chi-sam-mac/)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* M. Nissim for mentoring my project giving me guidance and tips
