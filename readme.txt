Installation
conda create -n foci-counter -f conda_env.yaml

Run Apps
import AppNucleusSegmentation as app
import tkinter as tk
root = tk.Tk()
app = app.NucleusSegmentation(root)
app.run()

import AppFociSegmentation as app
import tkinter as tk
root = tk.Tk()
app = app.FociSegmentation(root)
app.run()
