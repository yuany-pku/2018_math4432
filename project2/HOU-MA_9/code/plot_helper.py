# ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

from matplotlib import pyplot as plt
import numpy as np

def gridsearch_vis(clf, x_label, x_range):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearch Cross Validation")

    plt.xlabel(x_label)
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(min(x_range), max(x_range))
    ax.set_ylim(min(min(clf.cv_results_['mean_test_score']), min(clf.cv_results_['mean_train_score']))*0.9,
                max(max(clf.cv_results_['mean_test_score']), max(clf.cv_results_['mean_train_score']))*1.1)

    # Get the regular numpy array from the MaskedArray
    idx = []
    for j in x_range:
        idx.append([i for i, x in enumerate(clf.cv_results_['params']) if x[x_label] == j])
    X_axis = np.asarray(x_range)

    scorer = 'score'
    color = 'k'
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = np.asarray([np.max(clf.cv_results_['mean_%s_%s' % (sample, scorer)][i]) for i in idx])
        sample_score_std = np.asarray([np.max(clf.cv_results_['std_%s_%s' % (sample, scorer)][i]) for i in idx])
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.argwhere(X_axis==clf.cv_results_['param_'+x_label][np.argmin(clf.cv_results_['rank_test_score'])])[0][0]
    best_score = np.max(clf.cv_results_['mean_test_%s' % scorer])

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()