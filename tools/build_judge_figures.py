"""Render the pipeline and comparisons from public aggregate inputs only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/judge-guide'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'physiosleep-judge-guide-v1'})


def save(fig,name):
    for ext in ('png','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)


def comparisons():
    rows=json.loads((ROOT/'literature/workbook-comparison.json').read_text(encoding='utf8'))['experiments']
    fig,axes=plt.subplots(1,2,figsize=(17,10))
    for ax,group,title in zip(axes,['Small Sleep-EDF protocols','Expanded Sleep-EDF protocols'],['Small EDF cohorts · 18 workbook rows','Expanded EDF cohorts · 13 workbook rows']):
        subset=[r for r in rows if r['group']==group]
        for i,row in enumerate(subset):
            if row['macro_f1_percent'] is not None:
                ax.scatter(row['macro_f1_percent'],i,s=42,color='#286f8b')
                ax.text(row['macro_f1_percent']+.12,i,f"{row['macro_f1_percent']:g}",va='center',fontsize=9)
        ax.set_yticks(range(len(subset)),[f"{r['paper_id']} · {r['model']}" for r in subset],fontsize=9)
        ax.invert_yaxis();ax.set_xlim(70,87);ax.set_xlabel('Reported Macro-F1 (%)');ax.set_title(title,loc='left',pad=14)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Published-model context from the supplied workbook',fontsize=21,x=.03,ha='left')
    fig.text(.03,.93,'Source order retained; protocols differ within each panel. Values are workbook transcriptions, not independently verified paper tables.',fontsize=11)
    fig.text(.03,.025,'No PhysioSleep reference line or cross-study win/loss is computed. All 56 rows, source URLs, modalities and caveats are in literature/COMPARISON.md.',fontsize=10)
    fig.subplots_adjust(top=.86,bottom=.09,left=.23,right=.97,wspace=1.1)
    save(fig,'literature_context')
    local=json.loads((OUT/'local-comparison.json').read_text(encoding='utf8'))['models']
    fig,ax=plt.subplots(figsize=(12,6))
    for i,row in enumerate(local):
        ax.scatter(row['macro_f1'],i,s=65,color='#a55372' if row['name']=='Transition decoder' else '#23658c')
        ax.text(row['macro_f1']+.001,i,f"{row['macro_f1']:.6f}",va='center',fontsize=10)
    control=next(r['macro_f1'] for r in local if r['name']=='EEG + EOG · seed 17')
    ax.axvline(control+.02,ls='--',color='#b75b24');ax.text(control+.0205,-.45,'Control + 0.02',fontsize=10,color='#a34e1d')
    ax.set_yticks(range(len(local)),[r['name'] for r in local]);ax.invert_yaxis();ax.set_xlim(.71,.825)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Pooled fixed-five-class Macro-F1 (fraction)')
    ax.set_title('Actual local development results · 119 recordings / 60 people',loc='left',pad=22,fontsize=16)
    fig.text(.02,.015,'Same participant folds and valid-epoch denominator; inputs differ. Adaptive development, no audit. Dashed target is relative to one control only.',fontsize=9)
    fig.subplots_adjust(left=.29,bottom=.14,top=.89,right=.98);save(fig,'local_comparison')


def pipeline():
    fig,ax=plt.subplots(figsize=(15,8));ax.set_xlim(0,15);ax.set_ylim(0,8);ax.axis('off')
    def box(x,y,w,h,title,body,color='#e9f2f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='#7593a3',linewidth=1))
        ax.text(x+.18,y+h-.28,title,fontsize=12,weight='bold',va='top',color='#173449')
        ax.text(x+.18,y+h-.68,body,fontsize=10,va='top',linespacing=1.5,color='#173449')
    def arrow(start,end):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#365d73','lw':1.7})
    ax.text(.15,7.65,'PhysioSleep · what the reported pipeline actually does',fontsize=21,weight='bold')
    ax.text(.15,7.18,'Offline development pipeline. Reference annotations train and evaluate; they are never an inference input.',fontsize=12,color='#526577')
    box(.2,4.8,3,1.8,'1  Immutable source','197 PSG / Hypnogram pairs\n100 participants; native rates\nHash, timing, channel checks')
    box(4,4.8,4.4,1.8,'2  Participant partition','Development 60: five 48 / 12 folds\nOriginal A 20 + B 20: exploratory\nAll nights stay with their person')
    box(9.2,4.8,5.3,1.8,'3  Fold-training only','Native physiological features + LightGBM\nSeeds 17 / 43 / 101; train-only priors\nRecipe selection uses development only')
    arrow((3.3,5.7),(3.9,5.7));arrow((8.5,5.7),(9.1,5.7))
    box(.2,2.1,4.2,1.8,'4  Held-person PSG inference','EEG + EOG → features → classifiers\nMean seed probabilities → decoder\nOutput for every complete 30-s epoch')
    box(5.1,2.1,4.5,1.8,'5  Separate reference evaluation','Original grid + fixed validity mask\nPooled W / N1 / N2 / N3 / REM counts\nSaved outputs, hashes and verification')
    box(10.3,2.1,4.2,1.8,'6  Summary agreement','Predicted vs reference summaries\nTST / within-SPT WASO / score\n61 windows / 40 people; targets unmet','#fbf0e8')
    arrow((4.5,3),(5,3));arrow((9.7,3),(10.2,3))
    ax.plot([11.7,11.7,2.3],[4.7,4.5,4.5],color='#365d73',lw=1.7)
    arrow((2.3,4.5),(2.3,4.0))
    ax.text(4.1,4.1,'Frozen fold models + priors feed step 4; held-person labels feed step 5 only.',fontsize=10,color='#526577')
    box(.2,.1,8.1,1.1,'Future confirmation · NOT_RUN','Complete baselines + readiness → fresh confirmation people → unchanged gates','#eef0f4')
    box(8.9,.1,5.6,1.1,'Separate hardware concept · UNVALIDATED','Proposed 250-Hz acquisition; no device supplied these data','#eef0f4')
    save(fig,'pipeline')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);comparisons();pipeline()
