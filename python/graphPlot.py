# %%
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axisartist.axislines import SubplotZero

def drawEdgesAndLabels(options, g, list):
    nx.drawing.draw_networkx_edges(g, edgelist=list, arrows=True, **options)
    nx.drawing.draw_networkx_edge_labels(g, edgelist=list, **options)

def drawGraph(g, font='dejavu sans', arrow='->', layoutG=None):
    if not layoutG:
        layoutG = g

    sharedOpt = {
        'pos': nx.drawing.nx_agraph.graphviz_layout(layoutG, prog='dot'),
        'node_size': 2000
    }

    nodeOpt = {
        **sharedOpt,
        'node_color': 'white',
        'node_shape': "o",
        #         'clip_on': False,
        #         'Rotate': False,
        'font_family': font
    }
    arrowHeadOpt = {
        **sharedOpt,
        'node_shape': "s",
        'edge_labels': nx.get_edge_attributes(g, 'text'),
        'width': 2,
        'arrowstyle': 'wedge',
        'arrowsize': 30
    }
    arrowTailOpt = {
        **arrowHeadOpt,
        'arrowstyle': arrow
    }

    with plt.xkcd():
        fig, ax = plt.subplots(1)
        ax.axis('off')

        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)
        plt.xticks([])
        plt.yticks([])

        ax.set_ylabel('go up')
        ax.set_xlabel('go right')

        for s in ['right', 'top']:
            ax.axis[s].set_visible(False)

        for s in ['bottom', 'left']:
            ax.axis[s].set_axisline_style("->")

        nx.drawing.draw_networkx_nodes(g, **nodeOpt)
        nx.drawing.draw_networkx_labels(g, **nodeOpt)

        edges = g.edges.data()

        heads = [e for e in edges if ('wedge' in e[2])]
        drawEdgesAndLabels(arrowHeadOpt, g, heads)

        tails = [e for e in edges if ('wedge' not in e[2])]
        drawEdgesAndLabels(arrowTailOpt, g, tails)
