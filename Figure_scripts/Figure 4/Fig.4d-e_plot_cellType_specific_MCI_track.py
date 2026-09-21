import numpy as np
import pandas as pd
import cooler

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from itertools import combinations
import pybedtools

from genomic_regions import as_region

import math
import re
import matplotlib.pyplot as plt
import networkx as nx
import random

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

## FAN-C track visualization.

import numpy as np
import pandas as pd
import datatable as dt
import gc

import seaborn as sns

import matplotlib.pyplot as plt
sns.set_color_codes("pastel")
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Arc, Polygon, Rectangle
from intervaltree import Interval, IntervalTree
import pybedtools
import tempfile
import sys, os
import math

import fanc
import fanc.plotting as fancplot
from genomic_regions import GenomicRegion, as_region

def get_region(region):
    chrom, pos = region.split(':')
    start, end = pos.split('-')
    return chrom, int(start), int(end)

class InputError(Exception):
    "Exception raised for errors in the input."
    pass

def process_link_file(link_file, plot_regions):

    if plot_regions is None:
        file_to_open = link_file
    else:
        plot_regions_adapted = [(chrom, 0, int(3e8)) for chrom, __, __ in plot_regions]
        file_to_open = temp_file_from_intersect(link_file, plot_regions_adapted)

    interval_tree = {}
    line_number = 0
    with open(file_to_open, 'r') as f:
        for line in f:
            line_number += 1
            if line.startswith('#'):
                continue
            try:
                chrom1, start1, end1, chrom2, start2, end2 = line.strip().split('\t')[:6]
            except Exception as detail:
                raise InputError('File not valid. The file format should be'
                                f'chrom1 start1 end1 chrom2 start2 end2\n Error: {detail}\n'
                                f' in line\n {line}')
            if chrom1 != chrom2:
                continue

            start1, end1, start2, end2 = int(start1), int(end1), int(start2), int(end2)
            
            assert start1 <= end1, f'Error in line #{line_number}, end1 larger than start1 in {line}'
            assert start2 <= end2, f'Error in line #{line_number}, end2 larger than start2 in {line}'

            if chrom1 not in interval_tree:
                interval_tree[chrom1] = IntervalTree()
            
            if start2 < start1:
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            
            mid1 = (start1 + end1) / 2
            mid2 = (start2 + end2) / 2
            interval_tree[chrom1].add(Interval(mid1, mid2, [start1, end1, start2, end2]))
    return interval_tree

def temp_file_from_intersect(filename, plot_regions=None, around_region=0):
    """
    intersect filename with the plot_regions +/- around_region.
    :param filename: string file name.
    :param plot_regions: a list of tuple [(chrom1, start1, end1), (chrom2, start2, end2)]
    :param around_region: integer with the bo to extend the plot_region.
    :return: tmoporary file with the intersection.
    """
    file_to_open = filename
    if plot_regions is not None:
        original_file = pybedtools.BedTool(filename)
        plot_regions_ext = [(chrom, int(max(1, start-around_region)), int(end+around_region)) for chrom, start, end in plot_regions]
        ### overlap both version of chromosome name (with/without chr)
        plot_regions_as_bed = '\n'.join([f'{chrom}\t{start}\t{end}\n{change_chrom_names(chrom)}\t{start}\t{end}' for chrom, start, end in plot_regions_ext])
        regions = pybedtools.BedTool(plot_regions_as_bed, from_string=True)
        temporary_file = tempfile.NamedTemporaryFile(delete=False)
        sys.stderr = open(temporary_file.name, 'w')

        try:
            file_to_open = original_file.intersect(regions, wa=True, u=True).fn
            # print(file_to_open)
        except pybedtools.helpers.BEDToolsError:
            file_to_open = filename
            # print(f'===>{file_to_open}')
        sys.stderr.close()
        sys.stderr = sys.__stderr__
        with open(temporary_file.name, 'r') as f:
            temd_std_error = f.readlines()
        os.remove(temporary_file.name)
        error_lines = [line for line in temd_std_error if 'error' in line.lower()]
        if len(error_lines) >0:
            error_lines_printable = '\n'.join(error_lines)
            sys.stderr.write('Bedtools intersect raised an error:\n'
                             f"{error_lines_printable}\n"
                             "Will not use bedtools.\n")
            file_to_open = filename
    return file_to_open


def change_chrom_names(chrom):
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    else:
        chrom = 'chr' + chrom

    return chrom


def plot_arcs(ax, interval, color='#C5479B', compact_arcs_level=1):
    width = interval.end - interval.begin
    if compact_arcs_level == 1: ## the height is propotional to the square root of the distance.
        half_height = np.sqrt(width)
    elif compact_arcs_level == 2:  ## the height is the same for all distance.
        half_height = 1000
    else:
        half_height = width
    center = interval.end - width / 2

    ax.add_patch(Arc((center, 0), width, 2*half_height, 
                      0, 0, 180, color=color, alpha=1,
                      linewidth=1, ls='solid'))


def check_chrom_str_bytes(iteratable_obj, p_obj):

    def to_string(s):
        """
        This takes care of python2/3 differences
        """
        if isinstance(s, str):
            return s
        if isinstance(s, bytes):
            assert(sys.version_info[0] != 2)
        #        if sys.version_info[0] == 2:
        #            return str(s)
            return s.decode('ascii')
        if isinstance(s, list):
            return [to_string(x) for x in s]
        return s

    def to_bytes(s):
        """
        Like toString, but for functions requiring bytes in python3
        """
        assert(sys.version_info[0] != 2)
    #    if sys.version_info[0] == 2:
    #        return s
        if isinstance(s, bytes):
            return s
        if isinstance(s, str):
            return bytes(s, 'ascii')
        if isinstance(s, list):
            return [to_bytes(x) for x in s]
        return s
    # determine type
    if isinstance(p_obj, list) and len(p_obj) > 0:
        type_ = type(p_obj[0])
    else:
        type_ = type(p_obj)
    if not isinstance(type(next(iter(iteratable_obj))), type_):
        if type(next(iter(iteratable_obj))) is str:
            p_obj = to_string(p_obj)
        elif type(next(iter(iteratable_obj))) in [bytes, np.bytes_]:
            p_obj = to_bytes(p_obj)
    return p_obj


def plot_loop(ax, interval_tree, region_chrom, region_start, region_end, ylabel=''):
    region_chrom = check_chrom_str_bytes(interval_tree, region_chrom)
    arcs_in_region = sorted(interval_tree[region_chrom][region_start:region_end])

    ## get ylim.
    if arcs_in_region:
        yvalues = [np.sqrt(interval.end - interval.begin) for interval in arcs_in_region ]
        ymax = math.ceil(np.percentile(yvalues, 90) / 100 + 1) * 100

        for idx, interval in enumerate(arcs_in_region):
            if interval.begin < region_start and interval.end >region_end:
                continue
            plot_arcs(ax, interval, compact_arcs_level=1)
    else:
        ymax = 0

    ax.set_xlim([region_start, region_end])
    Xtick = list( np.linspace(region_start, region_end, 5 ,endpoint=True) )
    Xtick_label = [ "%.3f"%(i/10**6) for i in Xtick ]
    ax.set_xticks(Xtick)
    # ax.set_ylim([ymax, 0])
    ax.set_ylim([0, ymax])
    ax.set_yticks([])
    ax.set_ylabel(ylabel)
    ## remove frame border.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

def FancTrackLinePlot(filename, color, label, fill=True, ylabel=None, ylim=None, ax=None):
    track = fanc.load(filename)
    hp = fancplot.LinePlot(track, fill=fill, style='mid', colors=[color], labels=[label], ylabel=ylabel, ylim=ylim, n_yticks=2, ax=ax)
    return hp

def FancTrackBarPlot(filename, color, label, ylabel=None, ylim=None, ax=None):
    track = fanc.load(filename)
    hp = fancplot.BarPlot(track, style='step', colors=[color], labels=[label], ylabel=ylabel, ylim=ylim, n_yticks=2, ax=ax)
    return hp


def FancTrackRegionPlot(filename, color, ylabel=None, ax=None):
    track = fanc.load(filename)
    hp = fancplot.GenomicFeaturePlot(track, color=color, ylabel=ylabel, ax=ax)
    return hp


#### 可视化

## 1. GM12878 Hi-C heatmap
## 2. K562 Hi-C heatmap
## 3. predicted MCI.
## 4. GM12878 H3K4me1
## 5. K562 H3K4me1
## 6. GM12878 H3K27ac
## 7. K562 H3K27ac
## 8. GM12878 H3K4me3
## 9. K562 H3K4me3

# 1264	1278	1303	39	chr6_6320000-6325000	chr6_6390000-6395000	chr6_6515000-6520000	0.527711	0.035552375	F13A1		

def plot_candidate_MCI_track(region='chr6:6000000-7000000', 
                            node_bins_list=['chr6_6320000-6325000', 'chr6_6390000-6395000', 'chr6_6515000-6520000'],
                            specific_celltype='GM12878'):

    # region = as_region('chr6:6000000-7000000')
    region = as_region(region)
    hic_region = f'{region.chromosome.replace("chr", "")}:{(region.start/1000000):.2f}mb-{(region.end/1000000):.2f}mb'  ## hic_region = '1:100mb-230mb'
    fanc_region = f'{region.chromosome}:{(region.start/1000000):.2f}mb-{(region.end/1000000):.2f}mb'

    def convert_chrom_region(region):
        chrom, start, end = re.split(':|-|_', region)
        return f'{chrom}:{start}-{end}'

    fig, axes = plt.subplots(12, 1, figsize=(6, 8), constrained_layout=True,
                            gridspec_kw={'height_ratios': [0.1, 0.1, 0.01, 
                                                            0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
                                                            0.5]})
    axes = axes.flatten()

    ## set heatmap color.
    cmap_classic = LinearSegmentedColormap.from_list('normal', [(1, 1, 1), (1, 0, 0)], N=1000)  # white -> red
    cmap_diff = LinearSegmentedColormap.from_list('normal', [(0,0,1), (1,1,1), (1,0,0)], N=1000)  ## colormap: blue -> red
    cmap_oe = LinearSegmentedColormap.from_list('normal', ['#1f2d5f', '#6d93bf', '#edf0f2', '#dba786', '#520c1c'], N=1000) 

    ## 1. GM12878 Hi-C OE heatmap
    hic_file_dir = '/data/xujs/Project/HiPoreC_Promoter_Result/Result/2022-09-27_HiPoreC_to_cool_hic_files'
    gm12878_hic_filename = f'{hic_file_dir}/GM12878.Merge.HiPoreC.hic@5kb'
    gm12878_hic = fanc.load(gm12878_hic_filename)
    # HiC_plot = fancplot.TriangularMatrixPlot(hic, vmin=0, vmax=10000, max_dist=region.end - region.start, colormap=cmap_diff, show_colorbar=True, ax=axes[0])
    HiC_plot = fancplot.TriangularMatrixPlot(gm12878_hic, vmin=0, vmax=3, oe=True, max_dist='0.5mb', colormap=cmap_oe, show_colorbar=False, ax=axes[0])
    HiC_plot.plot(hic_region)
    axes[0].set_ylabel('GM12878 Hi-C', rotation=0, x=-0.1, y=0.3)

    ## 2. K562 Hi-C OE heatmap
    k562_hic_filename = f'{hic_file_dir}/K562.Merge.HiPoreC.hic@5kb'
    k562_hic = fanc.load(k562_hic_filename)
    HiC_plot = fancplot.TriangularMatrixPlot(k562_hic, vmin=0, vmax=3, oe=True, max_dist='0.5mb', colormap=cmap_oe, show_colorbar=False, ax=axes[1])
    HiC_plot.plot(hic_region)
    axes[1].set_ylabel('K562 Hi-C', rotation=0, x=-0.1, y=0.3)

    ## 3. H2P predicted 3-way MCI.
    # node_regions = ['chr6_6320000-6325000', 'chr6_6390000-6395000', 'chr6_6515000-6520000']
    node_regions = node_bins_list
    y = 0
    axes[2].plot([region.start, region.end], [y, y], color='#bebebe', linewidth=1)  ##灰色横线
    for node_region in node_regions:
        # bin = node2bin[node]
        node_region = as_region(convert_chrom_region(node_region))
        x = node_region.center
        axes[2].plot(x, y, 'o', color='black', markersize=3)

    Xtick = list( np.linspace(region.start, region.end, 5, endpoint=True) )
    Xtick_label = [ "%.3f"%(i/10**6) for i in list( np.linspace(region.start, region.end, 5 ,endpoint=True) ) ]
    axes[2].set_xlim([region.start, region.end])
    axes[2].set_ylim([-0.01, 0.01])
    axes[2].set_xticks(Xtick)
    axes[2].set_xticklabels(Xtick_label)
    axes[2].set_yticks([])
    axes[2].set_ylabel('MCI          ', rotation=0, x=-0.1, y=0.3)
    axes[2].spines['left'].set_visible(False)

    ## 4. GM12878 H3K4me1
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/GM12878/Histone-ChIP-seq/bigwig/GM12878.H3K4me1.ENCFF564KBE.GRCh38.bigWig'
    p4 = FancTrackLinePlot(filename, '#af2125', 'H3K4me1', True, 'H3K4me1', ylim=(0,20), ax=axes[3])
    p4.plot(fanc_region)
    axes[3].set_ylabel('GM12878 H3K4me1                      ', rotation=0, x=-0.1, y=0.3)
    ## 5. K562 H3K4me1
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/K562/Histone-ChIP-seq/bigwig/K562.H3K4me1.ENCFF457URZ.GRCh38.bigWig'
    p5 = FancTrackLinePlot(filename, '#af2125', 'H3K4me1', True, 'H3K4me1', ylim=(0,20), ax=axes[4])
    p5.plot(fanc_region)
    axes[4].set_ylabel('K562 H3K4me1                     ', rotation=0, x=-0.1, y=0.3)
    ## 6. GM12878 H3K27ac
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/GM12878/Histone-ChIP-seq/bigwig/GM12878.H3K27ac.ENCFF469WVA.GRCh38.bigWig'
    p6 = FancTrackLinePlot(filename, '#c96728', 'H3K27ac', True, 'H3K27ac', ylim=(0,20), ax=axes[5])
    p6.plot(fanc_region)
    axes[5].set_ylabel('GM12878 H3K27ac                      ', rotation=0, x=-0.1, y=0.3)
    ## 7. K562 H3K27ac
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/K562/Histone-ChIP-seq/bigwig/K562.H3K27ac.ENCFF465GBD.GRCh38.bigWig'
    p7 = FancTrackLinePlot(filename, '#c96728', 'H3K27ac', True, 'H3K27ac', ylim=(0,20), ax=axes[6])
    p7.plot(fanc_region)
    axes[6].set_ylabel('K562 H3K27ac                    ', rotation=0, x=-0.1, y=0.3)
    ## 8. GM12878 H3K4me3
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/GM12878/Histone-ChIP-seq/bigwig/GM12878.H3K4me3.ENCFF287HAO.GRCh38.bigWig'
    p8 = FancTrackLinePlot(filename, '#5e58a4', 'H3K4me3', True, 'H3K4me3', ylim=(0,10), ax=axes[7])
    p8.plot(fanc_region)
    axes[7].set_ylabel('GM12878 H3K4me3                   ', rotation=0, x=-0.1, y=0.3)
    ## 9. K562 H3K4me3
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/K562/Histone-ChIP-seq/bigwig/K562.H3K4me3.ENCFF405ZDL.GRCh38.bigWig'
    p9 = FancTrackLinePlot(filename, '#5e58a4', 'H3K4me3', True, 'H3K4me3', ylim=(0,10), ax=axes[8])
    p9.plot(fanc_region)
    axes[8].set_ylabel('K562 H3K4me3                  ', rotation=0, x=-0.1, y=0.3)
    ## 10. GM12878 RNA-seq
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/GM12878/RNA-seq/GM12878.RNA-seq.hg38.ENCFF563OCX.RPKM.bw'
    p10 = FancTrackLinePlot(filename, '#6e276c', 'H3K4me3', True, 'RNA-seq', ylim=(0,100), ax=axes[9])
    p10.plot(fanc_region)
    axes[9].set_ylabel('GM12878 RNA-seq                  ', rotation=0, x=-0.1, y=0.3)
    ## 11. K562 RNA-seq
    filename = f'/data/xujs/Public_Data/ENCODE_Data/human/hg38/K562/RNA-seq/K562.RNA-seq.hg38.ENCFF517WTR.RPKM.bw'
    p11 = FancTrackLinePlot(filename, '#6e276c', 'H3K4me3', True, 'RNA-seq', ylim=(0,100), ax=axes[10])
    p11.plot(fanc_region)
    axes[10].set_ylabel('K562 RNA-seq                  ', rotation=0, x=-0.1, y=0.3)

    ## 12. Gene annotation
    gtf_filename = '/data/Public_Data/References/human/hg38/gencode.v38.protein_coding.annotation.gtf'
    p = fancplot.GenePlot(gtf_filename, group_by='gene_id', squash=True, label_field='gene_name',
                            color_forward='#ff5754', color_reverse='#01096d', show_arrows=True,
                        arrow_size=3, relative_marker_step=0.015, line_width=1, box_height=0.3, ax=axes[-1])
    p.plot(fanc_region)

    ## 13. 添加vertical vline
    # node_regions = ['chr6_6320000-6325000', 'chr6_6390000-6395000', 'chr6_6515000-6520000']
    vlines_list = [int(re.split('_|-', x)[1])+2500 for x in node_regions]

    for pos in vlines_list:
        con =  ConnectionPatch(xyA=(pos, axes[0].get_ylim()[1]), xyB=(pos, axes[-2].get_ylim()[0]),
                                    coordsA=axes[0].transData, coordsB=axes[-2].transData,
                                    axesA=axes[0],
                                    axesB=axes[-2], color='#8d8d8d', ls='dashed') # 'ls:line_style'
        fig.add_artist(con)

    for ax in axes[:-1]:
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.spines['right'].set_visible(False) ## 去除边框
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    Xticks = list(np.linspace(region.start, region.end, 5, endpoint=True))
    Xtick_label = [ "%.3f"%(i/10**6) for i in list( np.linspace(region.start, region.end, 5 ,endpoint=True) ) ]
    axes[0].set_title(region)
    axes[-1].set_xlim([region.start, region.end])
    axes[-1].set_xticks(Xticks)
    axes[-1].set_xticklabels(Xtick_label)
    axes[-1].set_xlabel(f'{region.chromosome}(Mb)')
    axes[-1].set_yticks([])
    axes[-1].spines['right'].set_visible(False) ## 去除边框
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['left'].set_visible(False)

    fig.savefig(f'{specific_celltype}_specific_{region}.track.pdf')

    return fig


def get_plot_region(bin_region_list):

    def convert_chrom_region(region):
        chrom, start, end = re.split(':|-|_', region)
        return f'{chrom}:{start}-{end}'
    
    bin_1 = bin_region_list[0]
    bin_2 = bin_region_list[-1]
    bin_1 = as_region(convert_chrom_region(bin_1))
    bin_2 = as_region(convert_chrom_region(bin_2))
    chrom = bin_1.chromosome
    bs = bin_1.start
    be = bin_2.end

    expand_width = int((be - bs) / 4)
    region = f'{chrom}:{bs-expand_width}-{be+expand_width}'

    return region

# 1654	1680	1722	68	chr3_48270000-48275000	chr3_48400000-48405000	chr3_48610000-48615000	0.8264086	0.09491877			UQCRC1
# 2426	2464	2494	68	chr3_112130000-112135000	chr3_112320000-112325000	chr3_112470000-112475000	0.690456	0.47856057	GCSAM		
# 1264	1278	1303	39	chr6_6320000-6325000	chr6_6390000-6395000	chr6_6515000-6520000	0.527711	0.035552375	F13A1		
# 1248	1428	1573	325	chr6_26240000-26245000	chr6_27140000-27145000	chr6_27865000-27870000	0.8174723	0.23453805			H1-5
# 2602	2624	2630	28	chr6_33010000-33015000	chr6_33120000-33125000	chr6_33150000-33155000	0.89270115	0.012651047	HLA-DOA		
# 3101	3109	3156	55	chr6_35505000-35510000	chr6_35545000-35550000	chr6_35780000-35785000	0.98107237	0.39481664			CLPSL1
# 3141	3159	3172	31	chr7_35705000-35710000	chr7_35795000-35800000	chr7_35860000-35865000	0.92337143	0.028597003		SEPTIN7	
# 915	992	1113	198	chr8_64575000-64580000	chr8_64960000-64965000	chr8_65565000-65570000	0.9027458	0.021458423	BHLHE22		

# region='chr6:6000000-7000000', node_bins_list=['chr6_6320000-6325000', 'chr6_6390000-6395000', 'chr6_6515000-6520000']

# ####### GM12878 specific first batch
# regions = ['chr3:48100000-48900000', 
#             'chr3:112000000-112500000',
#             'chr6:6000000-6700000',
#             'chr6:26210000-27890000',
#             'chr6:32900000-33300000',
#             'chr6:35200000-35900000',
#             'chr6:35400000-36000000',
#             'chr8:64530000-65600000',
#             'chr10:14400000-14900000', 
#             'chr10:74000000-75300000',
#             'chr11:118200000-118800000',
#             'chr13:49800000-51200000',
#             'chr14:103900000-105000000',
#             'chr17:59400000-61000000'
#             ]

# MCI_node_region_list = [
#             ['chr3_48270000-48275000',	    'chr3_48400000-48405000',	    'chr3_48610000-48615000'],
#             ['chr3_112130000-112135000',	'chr3_112320000-112325000', 	'chr3_112470000-112475000'],
#             ['chr6_6320000-6325000',	    'chr6_6390000-6395000',	      'chr6_6515000-6520000'],
#             ['chr6_26240000-26245000',	    'chr6_27140000-27145000',    'chr6_27865000-27870000'],
#             ['chr6_33010000-33015000',	    'chr6_33120000-33125000',    'chr6_33150000-33155000'],
#             ['chr6_35505000-35510000',	    'chr6_35545000-35550000',    'chr6_35780000-35785000'],
#             ['chr7_35705000-35710000',	    'chr7_35795000-35800000',    'chr7_35860000-35865000'],
#             ['chr8_64575000-64580000',	    'chr8_64960000-64965000',    'chr8_65565000-65570000']，
#             ['chr10_14605000-14610000',	'chr10_14665000-14670000',	'chr10_14775000-14780000'],
#             ['chr10_74275000-74280000',	'chr10_74740000-74745000',	'chr10_75095000-75100000'],
#             ['chr11_118275000-118280000', 'chr11_118355000-118360000', 'chr11_118555000-118560000'],
#             ['chr13_50135000-50140000',	'chr13_50815000-50820000',	'chr13_50905000-50910000'],
#             ['chr14_104075000-104080000', 'chr14_104135000-104140000', 'chr14_104665000-104670000'],
#             ['chr17_360000-365000', 'chr17_900000-905000', 'chr17_980000-985000'],
#             ['chr17_59705000-59710000', 'chr17_59735000-59740000', 'chr17_59910000-59915000']
#         ]
# for i in trange(len(regions)):
#     print(f'plotting {regions[i]}')
#     plot_candidate_MCI_track(region=regions[i],
#                             node_bins_list=MCI_node_region_list[i],
#                             specific_celltype='GM12878')

# ####### GM12878 specific second batch
# gm12878_specific_regions = [
#                             'chr17:82500000-83200000',
#                             'chr18:62900000-63700000',
#                             'chr18:62200000-63200000',
#                             'chr19:6000000-7000000',
#                             'chr19:6400000-7000000',
#                             'chr19:13000000-13400000',
#                             'chr19:17200000-18000000',
#                             'chr19:41800000-42600000',
#                             'chr22:38900000-39700000',
#                             'chr22:39000000-39800000',
#                             'chr22:41100000-41800000',
#                             'chr22:42300000-43200000',
#                         ]

# gm12878_specific_MCI_node_region_list = [
#             ['chr17_82815000-82820000', 'chr17_82840000-82845000', 'chr17_83105000-83110000'],
#             ['chr18_63105000-63110000', 'chr18_63370000-63375000', 'chr18_63420000-63425000'],
#             ['chr18_62590000-62595000', 'chr18_62710000-62715000', 'chr18_62985000-62990000'],
#             ['chr19_6175000-6180000', 'chr19_6320000-6325000', 'chr19_6885000-6890000'],
#             ['chr19_6605000-6610000', 'chr19_6790000-6795000', 'chr19_6850000-6855000'],
#             ['chr19_13150000-13155000', 'chr19_13210000-13215000', 'chr19_13245000-13250000'],
#             ['chr19_17635000-17640000', 'chr19_17820000-17825000', 'chr19_17850000-17855000'],
#             ['chr19_42130000-42135000', 'chr19_42220000-42225000', 'chr19_42305000-42310000'],
#             ['chr22_39150000-39155000', 'chr22_39320000-39325000', 'chr22_39510000-39515000'],
#             ['chr22_39320000-39325000', 'chr22_39380000-39385000', 'chr22_39510000-39515000'],
#             ['chr22_41400000-41405000', 'chr22_41550000-41555000', 'chr22_41590000-41595000'],
#             ['chr22_42635000-42640000', 'chr22_42785000-42790000', 'chr22_43090000-43095000']
#         ]

# for i in trange(len(gm12878_specific_regions)):
#     print(f'plotting {gm12878_specific_regions[i]}')
#     plot_candidate_MCI_track(region=gm12878_specific_regions[i],
#                             node_bins_list=gm12878_specific_MCI_node_region_list[i],
#                             specific_celltype='GM12878')


#### K562 specific
# k562_specific_regions = ['chr7:157780000-158900000', 
#                         'chr9:128000000-130100000',
#                         'chr9:127900000-130000000',
#                         'chr9:133000000-133600000',
#                         'chr10:70400000-70800000',
#                         'chr20:34500000-35500000'
#                         ]

# k562_specific_MCI_node_region_list = [
#             ['chr7_158705000-158710000',	'chr7_158740000-158745000',	'chr7_158870000-158875000'],
#             ['chr9_128205000-128210000',	'chr9_128265000-128270000',	'chr9_129700000-129705000'],
#             ['chr9_128205000-128210000',	'chr9_128680000-128685000',	'chr9_129705000-129710000'],
#             ['chr9_133210000-133215000',	'chr9_133260000-133265000',	'chr9_133330000-133335000'],
#             ['chr10_70565000-70570000',	'chr10_70670000-70675000', 'chr10_70735000-70740000'],
#             ['chr20_34980000-34985000',	'chr20_35225000-35230000', 'chr20_35265000-35270000']
#         ]

# #### K562 candidate MCI.
# df_candidate = pd.read_csv('K562_specific_MCI_candidate.csv')
# for index, row in tqdm(df_candidate.iterrows()):
#     mci_node_regions = [row['n1_bin'], row['n2_bin'], row['n3_bin']]
#     region = get_plot_region(mci_node_regions)
#     print(f'{region} ...')

#     out_file = f'K562_specific_{region}.track.pdf'
#     if os.path.exists(out_file):
#         continue

#     try:
#         plot_candidate_MCI_track(region=region,
#                                 node_bins_list=mci_node_regions,
#                                 specific_celltype='K562')
#     except:
#         print(region)


## 选择四个区域重新画
regions = ['chr3:112000000-112500000', 
            'chr6:6000000-6700000', 
            'chr7:128375000-128975000',
            'chr21:44300000-45300000',
        ]
mci_node_regions = [
    ['chr3_112130000-112135000',	'chr3_112320000-112325000',    'chr3_112470000-112475000'],
    ['chr6_6320000-6325000',	    'chr6_6390000-6395000',	       'chr6_6515000-6520000'],
    ['chr7_128475000-128480000',    'chr7_128635000-128640000',    'chr7_128870000-128875000'],
    ['chr21_44485000-44490000',     'chr21_44935000-44940000',     'chr21_45145000-45150000']

]

# plot_candidate_MCI_track(region=regions[0], node_bins_list=mci_node_regions[0], specific_celltype='GM12878')
# plot_candidate_MCI_track(region=regions[1], node_bins_list=mci_node_regions[1], specific_celltype='GM12878')
plot_candidate_MCI_track(region=regions[2], node_bins_list=mci_node_regions[2], specific_celltype='K562')
# plot_candidate_MCI_track(region=regions[3], node_bins_list=mci_node_regions[3], specific_celltype='K562')