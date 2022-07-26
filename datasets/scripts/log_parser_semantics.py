#!/usr/bin/python

import os
import sys
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
import numpy as np
import math

# Auxiliar Functions


def getStats(data, confidence=0.95):
    rows = data.shape[0]
    columns = data.shape[1]
    output = np.zeros((columns, 3), dtype=np.float)
    for row in range(columns):
        a = data[:, row]
        print(f'{a},{row}')
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        output[row, :] = [m, max(m-h, 0.0), m+h]
    return output


def precision(relevant, received):
    """Computes the precision of a reply.
    """
    receivedRelevants = 0.0
    for item in received:
        if item in relevant:
            receivedRelevants += 1
    return receivedRelevants/len(received)


def averagePrecision(relevant, received):
    """Computes the average precision of a reply.
    """
    result = 0.0
    for k in range(len(received)):
        if received[k] in relevant:
            result += precision(relevant, received[:k+1])
    return result/len(relevant)


def mean_average_precision(relevant, received):
    rv = 0.0
    if len(relevant) > 0:
      for i in range(len(relevant)):
        rv += averagePrecision(relevant[i], received[i])
      return rv/len(relevant)
    else:
      return 0.0


begining = '{"QueryId":'

logFile = "../logs/client_all_queries.log"

groups = {00: "M2M", 10: "E2M(1/1)", 11: "E2M(1/2)", 12: "E2M(2/2)",
          20: "U2M(1)", 21: "U2M(2)", 22: "U2M(3)", 23: "U2M(4)"}

sortedGroups = []
for key in sorted(groups.keys()):
    sortedGroups += [groups[key]]

variants = ["precision", "count"]
methods = ["jaccard", "cosine"]
submethods = ["string", "levenshtein", "semantic"]

columnCount = len(sortedGroups)

results = {}

count = 0
maxMissing = 0
missing = {}
for key in groups.keys():
    missing[key] = list(range(30))

# Starting Parsing
print("Starting Parsing")

performance = {'jaccard': {'string': {}, 'levenshtein': {}, 'semantic': {
}}, 'cosine': {'string': {}, 'levenshtein': {}, 'semantic': {}}}

for line in open(logFile):
    if begining in line:
        #ToDo: Improve
        count += 1
        strippedLine = line.strip().split(',"Services')
        line = begining+"\"" + strippedLine[0][len(begining):]+"\",\"Services"+strippedLine[1]
        print(f'{line}')

        # Parse string, Identify group and id
        jsonDecoded = json.loads(line)
        group = groups[int(int(jsonDecoded["QueryId"])/100)]
        groupI = sortedGroups.index(group)
        queryId = int(jsonDecoded["QueryId"]) % 100

        # Remove this query from the missing list
        #currentMissing = missing.get(int(jsonDecoded["QueryId"])/100, [])
        #if queryId in currentMissing:
        #    currentMissing.remove(queryId)
        #else:
        #    continue
        #missing[int(jsonDecoded["QueryId"])/100] = currentMissing

        relevant = [queryId]
        services = jsonDecoded["Services"]
        #print(f'Services:{services}')

        for variant in variants:
            for method in methods:
                for submethod in services[method]:
                    submethodI = submethods.index(submethod)
                    
                    received = services[method][submethod]
                    
                    #print(f'Group: {group}')

                    value = averagePrecision(
                        relevant, services[method][submethod])
                    if group not in performance[method][submethod]:
                        performance[method][submethod][group] = []
                    performance[method][submethod][group].append(
                        (relevant, received))

                    level0 = results.get(variant, {})
                    column = groupI
                    if variant == "precision":
                        initialPrecision = np.zeros((30, columnCount), dtype=float)
                        level1 = level0.get(method, {})
                        level2 = level1.get(submethod, initialPrecision)
                        row = queryId
                        level2[[row], [column]] = value
                    elif variant == "count":
                        initialCount = np.zeros(
                            (10, columnCount), dtype=float)
                        initialCount[[0]] = 1
                        level1 = level0.get(method, {})
                        level2 = level1.get(submethod, initialCount)
                        row = math.ceil(value*10)-1
                        if row < 0:
                            row = 0
                        level2[[row], [column]] += 1.0/30
                        level2[[0], [column]] -= 1.0/30
                    else:
                        raise Exception("Oops: How did we get here :-)")
                    level1[submethod] = level2
                    level0[method] = level1
                    results[variant] = level0

print(f'Count: {count}')


#for key in missing.keys():
#    maxMissing = max(maxMissing, len(missing[key]))
#    for value in missing[key]:
#        print(f'{str(key)},{str(value)}')

#if maxMissing > 0:
#    print(f'{count},{maxMissing}')


# compute the mean average precision
for method in methods:
    for submethod in submethods:
        for group_key in groups:
            if groups[group_key] in performance[method][submethod]:
              temp_list = performance[method][submethod][groups[group_key]]
            else:
              temp_list = []
            relevant, received = [], []
            for rel, rec in temp_list:
                relevant.append(rel)
                received.append(rec)
            value = mean_average_precision(relevant, received)
            print(f'{method}/{submethod}/{groups[group_key]} = {value}')


print()

# global mPA
for method in methods:
    for submethod in submethods:
        relevant, received = [], []
        for group_key in groups:
            if groups[group_key] in performance[method][submethod]:
              temp_list = performance[method][submethod][groups[group_key]]
            else:
              temp_list = []
            for rel, rec in temp_list:
                relevant.append(rel)
                received.append(rec)
        value = mean_average_precision(relevant, received)
        print(f'{method}/{submethod} = {value}')


# Make Graphics
pdf = PdfPages('../out/heatmaps.pdf')

# Plot it out
for variant in variants:
    for method in methods:
        data = []
        for i in range(len(submethods)):
            submethod = submethods[i]
            data = results[variant][method][submethod]

            plt.rc('font', size='9')
            plt.subplot(1, len(submethods), i+1)
            plt.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, vmin=0, vmax=1)
            plt.title(submethod.capitalize())
            # Format
            fig = plt.gcf()
            # get the axis
            ax = plt.gca()
            # turn off the frame
            # ax.set_frame_on(False)
            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
            for t in ax.xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in ax.yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            ax.grid(False)
            ax.set_xlim(0, data.shape[1])
            ax.set_xticklabels(sortedGroups, minor=False)
            plt.xticks(rotation=90)

            if i == 0:
                if variant == "precision":
                    plt.ylabel("Services")
                    precisionYLabels = range(data.shape[0])
                    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
                    ax.set_yticklabels(precisionYLabels, minor=False)
                elif variant == "count":
                    plt.ylabel("Average Precision")
                    precisionYLabels = np.linspace(0, 1, data.shape[0]+1)
                    ax.set_yticks(np.arange(data.shape[0]+1), minor=False)
                    ax.set_yticklabels(precisionYLabels, minor=False)
            else:
                ax.set_yticks([], minor=False)

            if i == len(submethods)-1:
                if variant == "precision":
                    plt.colorbar(label="Average Precision")
                    fig.set_size_inches(6, 6)
                if variant == "count":
                    plt.colorbar(label="Occurrences")
                    fig.set_size_inches(6, 3)

        # fig.suptitle(method.capitalize())
        subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=None)
        fig.tight_layout()
        pdf.savefig()
        plt.close()

pdf.close()


pdf = PdfPages('../out/boxplot.pdf')
medianprops = dict(linewidth=4)
for method in methods:
    data = []
    for i in range(len(submethods)):
        submethod = submethods[i]
        data = results["precision"][method][submethod]
        plt.subplot(1, len(submethods), i+1)
        plt.rc('font', size='9')
        plt.boxplot(data, labels=sortedGroups,
                    medianprops=medianprops, whis=[5, 95])
        plt.title(submethod.capitalize())
        fig = plt.gcf()
        ax = plt.gca()
        plt.xticks(rotation=90)
        if i == 0:
            plt.ylabel("Average Precision")
        if i == len(submethods)-1:
            fig.set_size_inches(6, 3)

    # fig.suptitle(method.capitalize())
    subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=None)
    fig.tight_layout()
    pdf.savefig()
    plt.close()

pdf.close()
