import sys

import matplotlib.pyplot as plt
import pandas as pd
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import XMLBIFWriter

if False:  # os.path.isfile("model.xml"):
    model = XMLBIFReader("model.xml").get_model()
else:

    df = pd.read_csv("data.csv")

    model = LinearGaussianBayesianNetwork([('pixel', 'fps')])
    XMLBIFWriter(model).write_xmlbif("model.xml")
    model.fit(df)

# Step 4: Access the learned CPDs (Continuous Gaussian distributions)
for cpd in model.get_cpds():
    print(cpd)

import pandas as pd
import seaborn as sns

states = ["pixel", "fps"]
X_samples = model.simulate(1000, 35)
X_df = pd.DataFrame(X_samples, columns=states)

sns.jointplot(x=X_df["pixel"], y=X_df["fps"], kind="kde", height=10, space=0, cmap="viridis")
plt.show()

sys.exit()

# Define the new sigmoid function with adjusted steepness
import numpy as np
from matplotlib import pyplot as plt

import utils

center = 25
k = 0.35

x = np.linspace(0, 50, 500)
# plt.figure(figsize=(10, 6))
plt.plot(x, utils.sigmoid(x, k, center), color='purple')
plt.axvline(center, color='gray', linestyle=':')
plt.grid()
plt.show()
