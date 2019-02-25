# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:45:40 2019

@author: HIWI
"""



for i in range(6):
    
    plt.figure(figsize=(12, 12))

    for chN, ch in enumerate(ch_col[25*i:25*(i+1)]):

        predictors=p_col[(t_W*f_W)*(chN+25*i):(t_W*f_W)*(chN+(25*i)+1)]

        X = df[predictors].fillna(0, inplace=False)
        X = np.array(X)
        y = df['task']

        rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
                  scoring='accuracy')
        rfecv.fit(X, y)

        ax = plt.subplot(5, 5, chN+1)
        ax.pcolor(1/rfecv.ranking_.reshape((f_W,t_W)), cmap='binary')

        ax.set_title(ch+', '+str(rfecv.n_features_)+', '+str(round(rfecv.grid_scores_.max(), 2)))

    plt.tight_layout()
    plt.savefig(folder+'classify_'+str(f_W*t_W)+'_'+str(i+1)+'.png', bbox_inches='tight')

