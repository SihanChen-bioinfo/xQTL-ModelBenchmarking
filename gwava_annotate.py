"""
 Copyright 2013 EMBL - European Bioinformatics Institute

 Licensed under the Apache License, Version 2.0 (the
 "License"); you may not use this file except in
 compliance with the License.  You may obtain a copy of
 the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, 
 software distributed under the License is distributed on 
 an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
 KIND, either express or implied. See the License for the 
 specific language governing permissions and limitations 
 under the License.
"""

import sys, os
from subprocess import call
import pybedtools
from pybedtools import BedTool
from tabix import Tabix
from pandas import *

# read the GWAVA_DIR from the environment, but default to the directory above where the script is located
GWAVA_DIR = os.getenv('GWAVA_DIR', '../')

# set the pybedtools temp directory
pybedtools.set_tempdir(GWAVA_DIR+'/tmp/')

def encode_feats(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    cols = open(af+'.cols', 'r').readline().strip().split(',')
    intersection = feats.intersect(v, wb=True)
    sort_cmd = 'sort -k1,1 -k2,2n -k3,3n %s -o %s' % (intersection.fn, intersection.fn)
    call(sort_cmd, shell=True)
    annots = intersection.groupby(g=[9,10,11,12], c=6, ops='freqdesc')
    results = {}
    for entry in annots:
        fs = entry[4].strip(',').split(',')
        results[entry.name] = Series({e[0]: int(e[1]) for e in [f.split(':') for f in fs]})
    df = DataFrame(results, index = cols)
    # transpose to turn feature types into columns, and turn all the NAs in to 0s
    return df.T.fillna(0)

def gene_regions(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    
    # first establish all the columns in the annotation file
    cols = set(f[4] for f in feats)

    results = {}

    intersection = v.intersect(feats, wb=True)

    if len(intersection) > 0:
        annots = intersection.groupby(g=[1,2,3,4], c=9, ops='collapse')

        for entry in annots:
            regions = {}
            for region in entry[4].split(','):  
                if region in regions:
                    regions[region] += 1
                else:
                    regions[region] = 1

            results[entry.name] = Series(regions)

    df = DataFrame(results, index = cols)

    return df.T.fillna(0)

def gerp(vf, af, name="gerp"):
    v = BedTool(vf)
    t = Tabix(af)

    results = {}

    for var in v:
        try:
            result = 0.0
            num = 0
            for res in t.query(var.chrom, var.start, var.end):
                result += float(res[4])
                num += 1
            if num > 0:
                results[var.name] = result/num
        except:
            pass

    return Series(results, name=name)

def gc_content(vf, fa, flank=50):
    v = BedTool(vf)
    flanks = v.slop(g=pybedtools.chromsizes('hg19'), b=flank)
    nc = flanks.nucleotide_content(fi=fa)
    results = dict([ (r.name, float(r[5])) for r in nc ])
    return Series(results, name="GC") 

def average_gerp(vf, af, flank=50):
    v = BedTool(vf)
    flanks = v.slop(g=pybedtools.chromsizes('hg19'), b=flank)
    return gerp(flanks.fn, af, name="avg_gerp")

def feat_dist(vf, af, name):
    v = BedTool(vf)
    a = BedTool(af)
    closest = v.closest(a, D="b")
    results = dict([ (r.name, int(r[len(r.fields)-1])) for r in closest ])
    return Series(results, name=name)

def motifs(vf, af):
    v = BedTool(vf)
    cpg = BedTool(af)
    overlap = v.intersect(cpg, wb=True)
    results = dict([ (r.name, 1) for r in overlap ])
    return Series(results, name="pwm")

def cpg_islands(vf, af):
    v = BedTool(vf)
    cpg = BedTool(af)
    overlap = v.intersect(cpg, wb=True)
    results = dict([ (r.name, 1) for r in overlap ])
    return Series(results, name="cpg_island")

def segmentations(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    results = {}
    intersection = v.intersect(feats, wb=True)
    if len(intersection) > 0:
        annots = intersection.groupby(g=[1,2,3,4], c=8, ops='collapse')
        for entry in annots: 
            results[entry.name] = Series(entry[4].split(',')).value_counts()

    names = {
        'CTCF': 'CTCF_REG', 
        'E':    'ENH', 
        'PF':   'TSS_FLANK', 
        'R':    'REP', 
        'T':    'TRAN', 
        'TSS':  'TSS', 
        'WE':   'WEAK_ENH'
    }

    return DataFrame(results, index=names.keys()).T.rename(columns=names)   

def dnase_fps(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    results = {}
    intersection = feats.intersect(v, wb=True)
    if len(intersection) > 0:
        sort_cmd = 'sort -k6,6 -k7,7n -k8,8n -k9,9 %s -o %s' % (intersection.fn, intersection.fn)
        call(sort_cmd, shell=True)
        annots = intersection.groupby(g=[6,7,8,9], c=4, ops='collapse')
        for entry in annots:
            cells = entry[4].split(',') 
            results[entry.name] = len(cells)

    return Series(results, name='dnase_fps')

def bound_motifs(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    intersection = feats.intersect(v, wb=True)
    results = {}
    if len(intersection) > 0:
        sort_cmd = 'sort -k6,6 -k7,7n -k8,8n -k9,9 %s -o %s' % (intersection.fn, intersection.fn)
        call(sort_cmd, shell=True)
        annots = intersection.groupby(g=[6,7,8,9], c=4, ops='collapse')
        for entry in annots:
            cells = entry[4].split(',') 
            results[entry.name] = len(cells)

    return Series(results, name='bound_motifs')

def snp_stats(vf, af, stat='avg_het', flank=500):
    v = BedTool(vf)
    feats = BedTool(af)
    flanks = v.slop(g=pybedtools.chromsizes('hg19'), b=flank)
    intersection = feats.intersect(flanks, wb=True)
    results = {}
    if len(intersection) > 0:
        sort_cmd = 'sort -k6,6 -k7,7n -k8,8n -k9,9 %s -o %s' % (intersection.fn, intersection.fn)
        call(sort_cmd, shell=True)
        annots = intersection.groupby(g=[6,7,8,9], c=5, ops='collapse')

        for entry in annots:
            rates = entry[4].split(',')
            tot = reduce(lambda x, y: x + float(y), rates, 0.)
            rate = tot / (flank * 2)
            results[entry.name] = rate
        
    return Series(results, name=stat)

def seq_context(vf, fa):
    v = BedTool(vf)
    flanks = v.slop(g=pybedtools.chromsizes('hg19'), b=1)
    nc = flanks.nucleotide_content(fi=fa, seq=True, pattern="CG", C=True)
    cpg_context = Series(dict([ (r.name, float(r[14])) for r in nc ]))
    nucleotide = Series(dict([ (r.name, r[13][1].upper()) for r in nc ]))
    results = {}
    for b in 'ACGT':
        results['seq_'+b] = (nucleotide == b).apply(float)

    results['in_cpg'] = cpg_context

    return DataFrame(results) 

def repeats(vf, af):
    v = BedTool(vf)
    feats = BedTool(af)
    intersection = v.intersect(feats, wb=True)
    results = {}
    if len(intersection) > 0:
        annots = intersection.groupby(g=[1,2,3,4], c=8, ops='collapse')
        for entry in annots:
            types = entry[4].split(',') 
            results[entry.name] = len(types)

    return Series(results, name='repeat')

def coordinates(vf):
    vs = BedTool(vf)
    pos = {}
    for v in vs: 
        pos[v.name] = {'chr': v.chrom, 'start': int(v.start), 'end': int(v.stop)}
    return DataFrame(pos).T

DIR = GWAVA_DIR+'/source_data/'
CPG = DIR+'ucsc/ucsc_cpg_islands.bed.gz'
TSS = DIR+'encode/Gencodev10_TSS_May2012.gff.gz'
GERP = DIR+'gerp/gerp_whole_genome.bed.gz'
HG19 = DIR+'hg19/hg19.fa'
MOTIFS = DIR+'ensembl/MotifFeatures.gff.gz'
SEGMENTS = DIR+'ensembl/segmentation.bed.gz'
DNASE_FPS = DIR+'encode/all.footprints.bed.gz'
BOUND_MOTIFS = DIR+'encode/bound_motifs.bed.gz'
GENE_REGIONS = DIR+'ensembl/gene_regions.bed.gz'
SPLICE_SITES = DIR+'ensembl/splice_sites.bed.gz'
HET_RATES = DIR+'1kg/het_rates.bed.gz'
DAF = DIR+'1kg/daf.bed.gz'
REPEATS = DIR+'ucsc/ucsc_repeats.bed.gz'
ENCODE_FEATS = DIR+'encode/encode_megamix.bed.gz'

def annotate(vf):
  
    df = coordinates(vf)

    annots = [
        encode_feats(vf, ENCODE_FEATS),
        motifs(vf, MOTIFS),
        cpg_islands(vf, CPG),
        average_gerp(vf, GERP),
        gerp(vf, GERP),
        feat_dist(vf, TSS, name='tss_dist'),  
        gc_content(vf, HG19),
        segmentations(vf, SEGMENTS),
        dnase_fps(vf, DNASE_FPS),
        bound_motifs(vf, BOUND_MOTIFS),
        gene_regions(vf, GENE_REGIONS),
        feat_dist(vf, SPLICE_SITES, name='ss_dist'),
        snp_stats(vf, HET_RATES, stat='avg_het'),
        snp_stats(vf, DAF, stat='avg_daf'),
        seq_context(vf, HG19),
        repeats(vf, REPEATS)
    ]

    # make sure everything is a DataFrame
    annots = [a if isinstance(a, DataFrame) else DataFrame({a.name: a}) for a in annots]
    
    # and join them all together
    df = df.join(annots)

    df = df.fillna(0)

    return df

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Usage:",sys.argv[0]," variants.bed output.csv class"
        sys.exit(0)

    vf = sys.argv[1]
    out = sys.argv[2]

    if len(sys.argv) == 4:
        cls = sys.argv[3]
    else:
        cls = 0

    df = annotate(vf)

    df['cls'] = int(cls)

    df.to_csv(out)

