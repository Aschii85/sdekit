import numpy as np
import plotly.express as px

from sdekit.process import MarkovProcess, GBMProcess, ItoProcess
from sdekit.simulation import simulate

processes: list[MarkovProcess] = [
    GBMProcess.from_equation('GBM', 'dX = (0.001 * sin(t) * X) * dt + (0.02 * X) * dWt'),
    # GBMProcess('GBM2', mu= 0.01, sigma=0.02),
    ItoProcess.from_equation('X0', 'dX = (0.001 * GBM + 0.001 * X) * dt + (0.01 * GBM + 0.02 * X) * dWt'),
    ItoProcess.from_equation('X1', 'dX = (0.02 * X) * dWt'),
]
s = simulate(
    method='euler',
    processes=processes,
    start_values=[1.0] * len(processes),
    time_steps=(range(1_000)),
    scenarios=10_000,
    wiener_correlations=np.array([[1, 0.5, 0.25],[0.5, 1, 0.5],[0.25, 0.5, 1]]),
)
print(s.values())
df = s.to_dataframe()
for p in processes:
    fig = px.line(
        df.reset_index(),
        x = 'time',
        y = p.name,
        color='scenario',
    )
    fig.show()
