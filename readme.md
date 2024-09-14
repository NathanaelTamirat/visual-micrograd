### Visual Micrograd
Inspired by Andrej Karpathy(the goat).

![vizualization](<action.png>)


In the interactive Demo there is a nice visaulization of the "action" during training, showing the exact computational graph. The full computational graph includes the entire dataset which is too much. So instead, here we pick a single datapoint to forward through the network and show that graph alone. In this case we're forwarding the origin (0,0) and a fictional label 0, running the forward pass and the loss, doing backward, and then showing the data/grads. The process to get this visualization working:

1. First run python `micrograd.py` and make sure it saves the `graph.svg`, which is the connectivity graph. You'll need to install graphviz if you don't have it. E.g. on my MacBook this is `brew install graphviz` followed by `pip install graphviz`.
2. Once we have the `graph.svg` file, we load it from the HTML. Because of cross-origin security issues when loading the svg from the file system, we can't just open up the HTML page directly and need to serve these files. The easiest way is to run a simple python webserver with `python -m http.server`. Then open up the localhost URL it gives you, and open the `micrograd.html` page. 

You'll see a really cool visualization of the training process.
