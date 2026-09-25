"""Plot public aggregate score errors; no participant or epoch inputs required."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=json.loads(args.summary.read_text(encoding='utf8'))
    if data['scope']!='descriptive_development' or data['score_recordings']!=61 or data['score_participants']!=40:
        raise ValueError('Expected frozen development score population')
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'physiosleep-score-extension-v1'})
    systems=['train_only_constant','eog','fpz_eeg','fpz_eeg_eog','transition']
    labels=['Train-only constant','EOG','EEG','EEG + EOG','Three-seed + transition']
    endpoints=[('score','Experimental score · points',5),('tst_minutes','TST · minutes',15),('waso_minutes','Within-SPT WASO · minutes',10)]
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
    for ax,(endpoint,title,target) in zip(axes,endpoints):
        for i,system in enumerate(systems):
            row=data['models'][system][endpoint]
            if row['status']!='DESCRIPTIVE_DEVELOPMENT':raise ValueError('Missing required score cannot be plotted as complete')
            lo,hi=row['cluster_ci95']['mae'];value=row['mae']
            color='#a55372' if system=='transition' else '#23658c'
            ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color=color,markersize=7)
        ax.axvline(target,ls='--',color='#b75b24',lw=1.5,label=f'Planned maximum MAE: {target}')
        ax.set_title(title,loc='left',fontsize=12,pad=30)
        ax.set_xlabel('Participant-balanced MAE')
        ax.set_xlim(left=0);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.text(0,1.02,f'Dashed line: planned maximum MAE {target}',transform=ax.transAxes,fontsize=9,color='#a34e1d')
    axes[0].set_yticks(range(5),labels);axes[0].invert_yaxis()
    fig.suptitle('Does the best staging pipeline improve score fidelity?',x=.035,ha='left',fontsize=19,fontweight='bold')
    fig.text(.035,.90,'61 eligible recording windows · 40 development participants · unchanged formula and eligibility',fontsize=11,color='#526577')
    fig.text(.035,.035,'Intervals: descriptive participant-bootstrap 95%, 2,000 common draws. All three new MAEs remain above their targets.\nDifferences versus EEG+EOG have intervals crossing zero; no confirmed improvement, audit or clinical claim.',fontsize=10,color='#526577')
    fig.subplots_adjust(left=.19,right=.99,bottom=.22,top=.78,wspace=.22)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':None})
    plt.close(fig)


if __name__=='__main__':main()
