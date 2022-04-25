from functools import partial
from DataGenerator import incrementalDrift, suddenDrift
import numpy as np
import pandas as pd


def scenarios():
    functList = []

    suddenDriftSmall = partial(suddenDrift, magnitude=1.5)
    suddenDriftLarge = partial(suddenDrift, magnitude=4.5)
    incrementalDriftSmall = partial(incrementalDrift, magnitude=1.5)
    incrementalDriftLarge = partial(incrementalDrift, magnitude=4.5)

    suddenDriftSmallImportant = partial(suddenDriftSmall, variables=0)
    suddenDriftsmallMedium = partial(suddenDriftSmall, variables=3)
    suddenDriftsmallUnimportant = partial(suddenDriftSmall, variables=6)

    suddenDriftLargeImportant = partial(suddenDriftLarge, variables=0)
    suddenDriftLargeMedium = partial(suddenDriftLarge, variables=3)
    suddenDrifLargeUnimportant = partial(suddenDriftLarge, variables=6)

    incrementalDriftSmallImportant = partial(
        incrementalDriftSmall, variables=0)
    incrementalDriftsmallMedium = partial(incrementalDriftSmall, variables=3)
    incrementalDriftsmallUnimportant = partial(
        incrementalDriftSmall, variables=6)

    incrementalDriftLargeImportant = partial(
        incrementalDriftLarge, variables=0)
    incrementalDriftLargeMedium = partial(incrementalDriftLarge, variables=3)
    incrementalDrifLargeUnimportant = partial(
        incrementalDriftLarge, variables=6)

    ############################################################################

    suddenDriftSmallImportantO = partial(suddenDriftSmallImportant, time=364)
    functList.append(suddenDriftSmallImportantO)
    suddenDriftSmallImportantH = partial(suddenDriftSmallImportant, time=912)
    functList.append(suddenDriftSmallImportantH)
    suddenDriftSmallImportantU = partial(suddenDriftSmallImportant, time=1100)
    functList.append(suddenDriftSmallImportantU)

    suddenDriftsmallMediumO = partial(suddenDriftsmallMedium, time=364)
    functList.append(suddenDriftsmallMediumO)
    suddenDriftsmallMediumH = partial(suddenDriftsmallMedium, time=912)
    functList.append(suddenDriftsmallMediumH)
    suddenDriftsmallMediumU = partial(suddenDriftsmallMedium, time=1100)
    functList.append(suddenDriftsmallMediumU)

    suddenDriftsmallUnimportantO = partial(
        suddenDriftsmallUnimportant, time=364)
    functList.append(suddenDriftsmallUnimportantO)
    suddenDriftsmallUnimportantH = partial(
        suddenDriftsmallUnimportant, time=912)
    functList.append(suddenDriftsmallUnimportantH)
    suddenDriftsmallUnimportantU = partial(
        suddenDriftsmallUnimportant, time=1100)
    functList.append(suddenDriftsmallUnimportantU)

    suddenDriftLargeImportantO = partial(suddenDriftLargeImportant, time=364)
    functList.append(suddenDriftLargeImportantO)
    suddenDriftLargeImportantH = partial(suddenDriftLargeImportant, time=912)
    functList.append(suddenDriftLargeImportantH)
    suddenDriftLargeImportantU = partial(suddenDriftLargeImportant, time=1100)
    functList.append(suddenDriftLargeImportantU)

    suddenDriftLargeMediumO = partial(suddenDriftLargeMedium, time=364)
    functList.append(suddenDriftLargeMediumO)
    suddenDriftLargeMediumH = partial(suddenDriftLargeMedium, time=912)
    functList.append(suddenDriftLargeMediumH)
    suddenDriftLargeMediumU = partial(suddenDriftLargeMedium, time=1100)
    functList.append(suddenDriftLargeMediumU)

    suddenDrifLargeUnimportantO = partial(suddenDrifLargeUnimportant, time=364)
    functList.append(suddenDrifLargeUnimportantO)
    suddenDrifLargeUnimportantH = partial(suddenDrifLargeUnimportant, time=912)
    functList.append(suddenDrifLargeUnimportantH)
    suddenDrifLargeUnimportantU = partial(
        suddenDrifLargeUnimportant, time=1100)
    functList.append(suddenDrifLargeUnimportantU)

    incrementalDriftSmallImportantO = partial(
        incrementalDriftSmallImportant, time=[364, 729])
    functList.append(incrementalDriftSmallImportantO)
    incrementalDriftSmallImportantH = partial(
        incrementalDriftSmallImportant, time=[912, 1277])
    functList.append(incrementalDriftSmallImportantH)
    incrementalDriftSmallImportantU = partial(
        incrementalDriftSmallImportant, time=[1100, 1459])
    functList.append(incrementalDriftSmallImportantU)

    incrementalDriftsmallMediumO = partial(
        incrementalDriftsmallMedium, time=[364, 729])
    functList.append(incrementalDriftsmallMediumO)
    incrementalDriftsmallMediumH = partial(
        incrementalDriftsmallMedium, time=[912, 1277])
    functList.append(incrementalDriftsmallMediumH)
    incrementalDriftsmallMediumU = partial(
        incrementalDriftsmallMedium, time=[1100, 1459])
    functList.append(incrementalDriftsmallMediumU)

    incrementalDriftsmallUnimportantO = partial(
        incrementalDriftsmallUnimportant, time=[364, 729])
    functList.append(incrementalDriftsmallUnimportantO)
    incrementalDriftsmallUnimportantH = partial(
        incrementalDriftsmallUnimportant, time=[912, 1277])
    functList.append(incrementalDriftsmallUnimportantH)
    incrementalDriftsmallUnimportantU = partial(
        incrementalDriftsmallUnimportant, time=[1100, 1459])
    functList.append(incrementalDriftsmallUnimportantU)

    incrementalDriftLargeImportantO = partial(
        incrementalDriftLargeImportant, time=[364, 729])
    functList.append(incrementalDriftLargeImportantO)
    incrementalDriftLargeImportantH = partial(
        incrementalDriftLargeImportant, time=[912, 1277])
    functList.append(incrementalDriftLargeImportantH)
    incrementalDriftLargeImportantU = partial(
        incrementalDriftLargeImportant, time=[1100, 1459])
    functList.append(incrementalDriftLargeImportantU)

    incrementalDriftLargeMediumO = partial(
        incrementalDriftLargeMedium, time=[364, 729])
    functList.append(incrementalDriftLargeMediumO)
    incrementalDriftLargeMediumH = partial(
        incrementalDriftLargeMedium, time=[912, 1277])
    functList.append(incrementalDriftLargeMediumH)
    incrementalDriftLargeMediumU = partial(
        incrementalDriftLargeMedium, time=[1100, 1459])
    functList.append(incrementalDriftLargeMediumU)

    incrementalDrifLargeUnimportantO = partial(
        incrementalDrifLargeUnimportant, time=[364, 729])
    functList.append(incrementalDrifLargeUnimportantO)
    incrementalDrifLargeUnimportantH = partial(
        incrementalDrifLargeUnimportant, time=[912, 1277])
    functList.append(incrementalDrifLargeUnimportantH)
    incrementalDrifLargeUnimportantU = partial(
        incrementalDrifLargeUnimportant, time=[1100, 1459])
    functList.append(incrementalDrifLargeUnimportantU)
    functList.append(lambda x: x)  # no drift

    ###########################################################################

    drop_none = None
    drop_important = 0
    drop_medium = 3
    drop_unimportant = 6
    drop_combination = [0, 3, 6]

    dropList = [drop_none, drop_important, drop_medium,
                drop_unimportant, drop_combination]

    ###########################################################################

    driftType = np.empty((len(functList)), dtype=object)
    driftType[0: 18] = 'Sudden Drift'
    driftType[18: 36] = 'Incremental Drift'
    driftType[36] = 'No Drift'

    driftMagnitude = np.empty((len(functList)), dtype=object)

    dMagnitude = np.empty(18, dtype=object)
    dMagnitude[0:9] = 'Small'
    dMagnitude[9:18] = 'Large'
    driftMagnitude[0:36] = np.tile(dMagnitude, 2)
    driftMagnitude[36] = 'No Drift'

    driftImportance = np.empty((len(functList)), dtype=object)

    dImportance = np.empty(9, dtype=object)
    dImportance[0:3] = 'Important'
    dImportance[3:6] = 'Medium'
    dImportance[6:9] = 'Unimportant'
    driftImportance[0:36] = np.tile(dImportance, 4)
    driftImportance[36] = 'No Drift'

    times = np.empty(3, dtype=object)
    driftTime = np.empty((len(functList)), dtype=object)
    times[0] = 'Fully observed'
    times[1] = 'Half observed'
    times[2] = 'Unobserved'
    driftTime[0:36] = np.tile(times, 12)
    driftTime[36] = 'No Drift'

    ###########################################################################

    scenarioList = []
    for drop in dropList:
        for funct, type, magnitude, importance, time in zip(functList, driftType, driftMagnitude, driftImportance, driftTime):
            scenario = {'function': funct, 'Dropped variable': drop, 'Drift type': type,
                        'Drift magnitude': magnitude, 'Variable importance': importance, 'Drift time': time}
            scenarioList.append(scenario)

    return scenarioList
