import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.\
    event_accumulator import EventAccumulator

"""
Confidence interval plotting from Tensorboard log data.

Log directories must be in a pre defined form where method name is
seperated with a '-' symbol.

Example

--Log Directory
    --lstm-2022-1
        --events.out.tfevents.1
    --lstm-2022-2
        --events.out.tfevents.1
    --lstm-2022-3
        --events.out.tfevents.1
    --no_memory-2022-1
        --events.out.tfevents.4
    --no_memory-2022-2
        --events.out.tfevents.5
    --no_memory-2022-3
        --events.out.tfevents.6
    --oa_k-2022-1
        --events.out.tfevents.7
    --oa_k-2022-2
        --events.out.tfevents.8
    --oa_k-2022-3
        --events.out.tfevents.9


In this case, this method creates lists of methods
(where method names are lstm, no_memory and oa_k in the above example)
and this script calculates confidence intervals of same methods.

In the confidence interval calculation, since the array sizes of each
run might differ, so all of the arrays are extended with NaNs.
NaN values are not taken into consideration in calculation of
confidence interval at that time step.
Therefore only existing values does not affect the
confidence interval calculation.
"""


def getSRperEpisodeFromDirectory(dpath):
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname),
                         size_guidance={
                         event_accumulator.SCALARS: 0,
                         }).Reload() for dname in os.listdir(dpath)]

    keys = list(set([summary.path.replace(dpath, '').split('-')[0]
                     for summary in summary_iterators]))
    indexes = {k: [] for k in keys}
    values = {k: [] for k in keys}

    for summary in summary_iterators:
        experiment_name = summary.path.replace(dpath, '').split('-')[0]
        SRperEpisode = summary.Scalars('_tmaze/Success Ratio per episode')
        indexes[experiment_name].append(np.fromiter
                                        (map(lambda x: x.step, SRperEpisode),
                                         dtype=int))
        values[experiment_name].append(np.fromiter(
            map(lambda x: x.value, SRperEpisode), dtype=np.double))

    return indexes, values, keys


def savePlotSRWithCI(indexes, values, keys):
    fig, ax = plt.subplots()
    color = iter(cm.brg(np.linspace(0, 1, len(keys))))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Success Ratio Per Episode")
    lines = []

    for key in keys:
        indexesList = indexes[key]
        valuesList = values[key]
        numberOfExperiments = len(indexesList)
        maxLength = np.max(np.array
                           ([len(indexList) for indexList in indexesList]))
        x = np.arange(1, maxLength+1, 1)
        extendList = valuesList
        for i in range(len(valuesList)):
            extend = np.empty((maxLength,))
            extend[:] = np.nan
            extend[:valuesList[i].shape[0]] = valuesList[i]
            extendList[i] = extend

        y = np.stack(extendList, axis=0)
        ci = 1.96 * np.nanstd(y, axis=0)/np.sqrt((numberOfExperiments))
        mean_y = np.nanmean(y, axis=0)

        c = next(color)
        line, = ax.plot(x, mean_y, color=c)
        lines.append(line)
        ax.fill_between(x, (mean_y-ci), (mean_y+ci), color=c, alpha=.1)

    ax.legend(lines, keys)
    fig.savefig("results.pdf", format="pdf", bbox_inches="tight")


path = 'results/results_comp_arch/c_architectures_tb/'

indexes, values, keys = getSRperEpisodeFromDirectory(path)
print("Read from file has been completed.")
savePlotSRWithCI(indexes, values, keys)
print("Plots have been generated.")
