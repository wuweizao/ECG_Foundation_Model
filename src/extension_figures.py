"""Plots for predeclared corruption stress tests and entropy diagnostics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .report import style,save,COLORS,NAMES


def main():
    style()
    df = pd.read_csv('results/robustness.csv')
    for metric,name,title in [('macro_auroc','robustness','Macro AUROC under input corruption'),
                              ('mean_binary_entropy','uncertainty','Mean binary predictive entropy under input corruption')]:
        fig,axes = plt.subplots(1,3,figsize=(12,4.5),sharey=True)
        for ax,(kind,levels,labels) in zip(axes,[('gaussian',[20,10,5],['Clean','20 dB','10 dB','5 dB']),
                                               ('dropout',[1,3,6],['Clean','1','3','6']),
                                               ('wander',[.2,.5],['Clean','0.2 Hz','0.5 Hz'])]):
            for model,marker,ls in [('scratch','s','--'),('pretrained','o','-')]:
                part = df[df.model==model]
                values = [part[part.corruption=='clean'][metric].iloc[0]]
                values += [part[(part.corruption==kind)&(part.level==level)][metric].iloc[0] for level in levels]
                ax.plot(range(len(values)),values,marker=marker,ls=ls,c=COLORS[model],label=NAMES[model])
            ax.set_xticks(range(len(labels)),labels)
            ax.set_title({'gaussian':'Gaussian noise','dropout':'Dropped leads','wander':'Baseline wander'}[kind],loc='left',fontsize=11)
            ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Macro AUROC' if metric=='macro_auroc' else 'Binary entropy (nats)')
        axes[0].set_ylim(0,1 if metric=='macro_auroc' else .7)
        axes[0].legend(frameon=False,loc='lower left')
        fig.suptitle(title,x=.07,ha='left')
        fig.text(.07,.89,'PTB-XL fold 10 | 10% labels, seed 42 | paired post-preprocessing perturbations',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.85))
        save(fig,name)


if __name__ == '__main__':
    main()
