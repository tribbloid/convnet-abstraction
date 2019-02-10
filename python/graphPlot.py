# %%
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axisartist.axislines import SubplotZero

SIZE = [14, 10]


def drawEdgesAndLabels(
        options, g,
        edgeFactory=lambda g: nx.get_edge_attributes(g, 'text')
):
    _options = {
        **options,
        'edge_labels': edgeFactory(g)
    }
    _labelOptions = {
        **_options,
        'bbox': dict(alpha=0.05, color='w')
    }
    nx.drawing.draw_networkx_edges(g, arrows=True, **_options)
    nx.drawing.draw_networkx_edge_labels(g, **_labelOptions)


def drawGraph(g, layoutG=None, **kwargs):
    if not layoutG:
        layoutG = g

    edges = g.edges.data()
    tails = [e[0:2] for e in edges if ('wedge' in e[2])]
    tailGraph = g.edge_subgraph(tails)

    heads = [e[0:2] for e in edges if ('wedge' not in e[2])]
    headGraph = g.edge_subgraph(heads)

    defaultOpt = {
        'pos': nx.drawing.nx_agraph.graphviz_layout(layoutG, prog='dot'),
        'node_size': 2000,
        'arrowstyle': '->',
        'font': 'dejavu sans'
    }
    sharedOpt = {
        **defaultOpt,
        **kwargs
    }

    nodeOpt = {
        **sharedOpt,
        'node_color': 'white',
        'node_shape': "o",
        # 'font_family': 'dejavu sans'
    }

    arrowOpt = {
        **sharedOpt,
        'node_shape': "s",
        'width': 2,
        'edge_color': '#AAAAAA',
        'arrowsize': 30
    }

    arrowHeadOpt = {
        **arrowOpt
    }

    arrowTailOpt = {
        **arrowOpt,
        'arrowstyle': 'wedge'
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

        drawEdgesAndLabels(arrowHeadOpt, headGraph)
        drawEdgesAndLabels(arrowTailOpt, tailGraph)
