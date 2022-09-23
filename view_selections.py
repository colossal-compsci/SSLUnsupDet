import numpy as np
import matplotlib.pyplot as plt
"""
DIRECTORY = '19620917b'

D = np.load(f'../Watkins_Detection/PipelineV0/Inference/{DIRECTORY}/spectrogram.npy')
sb = np.load(f'../Watkins_Detection/PipelineV0/Inference/{DIRECTORY}/sbgram.npy')

fig,axs = plt.subplots(2, 1, sharex=True, figsize=(16, 18))
axs[0].imshow(D, aspect='auto')
axs[0].invert_yaxis()
axs[1].imshow(sb, aspect='auto')
axs[1].invert_yaxis()
plt.show()
"""

import plotly.express as px
import numpy as np
import pandas as pd

DIRECTORY = '19911021d'

D = np.load(f'../Watkins_Detection/PipelineV0/Inference/{DIRECTORY}/spectrogram.npy')
sb = np.load(f'../Watkins_Detection/PipelineV0/Inference/{DIRECTORY}/sbgram.npy')
df = pd.read_pickle(f'../Watkins_Detection/PipelineV0/Inference/{DIRECTORY}/distances.pkl')
"""USE THESE
#df = df[df.Alphas == 0.1]
#df = df[np.isclose(df.Betas, 0.9)]
#df = df[df.SmoothDurations == 5]
"""
df = df[np.isclose(df.Alphas, 0.1)]
print(df.head())
df = df[np.isclose(df.Betas, 0)]
df = df[df.SmoothDurations == 5]
d = df.Distances.iloc[0]
import matplotlib.pyplot as plt
#plt.plot(d)
#plt.show()

x = np.arange(0, len(d))
print(x.shape, x[-1])


np.random.seed(123)
e = np.random.randn(100000,3)  
df=pd.DataFrame(e, columns=['a','b','c'])

df['x'] = df.index
df_melt = pd.melt(df, id_vars="x", value_vars=df.columns[:-1])
#fig=px.line(df_melt, x="x", y="value",color="variable")
fig=px.line(d)

# Add range slider
fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                             type="linear")
)

fig.show()

fig = px.imshow(np.flipud(D), aspect='auto')
fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                             type="linear")
)

fig.show()
fig = px.imshow(np.flipud(sb), aspect='auto')
fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                             type="linear")
)

fig.show()
