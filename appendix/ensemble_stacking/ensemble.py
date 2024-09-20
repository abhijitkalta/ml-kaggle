'''remember that smaller models are also easier and faster to tune.
Ensembling is nothing but a combination of different models. The models can be
combined by their predictions/probabilities. The simplest way to combine models
would be just to do an average.
Ensemble Probabilities = (M1_proba + M2_proba + … + Mn_Proba) / n'''
# The first rule of ensembling is that you always create folds before starting with
#ensembling.
import numpy as np
def mean_predictions(probas):
    """
    Create mean predictions
    :param probas: 2-d array of probability values
    :return: mean probability
    """
    return np.mean(probas, axis=1)
def max_voting(preds):
    """
    Create mean predictions
    :param probas: 2-d array of prediction values
    :return: max voted predictions
    """
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)

''' Another way of combining multiple models is by
ranks of their probabilities. This type of combination works quite good when the
concerned metric is the area under curve as AUC is all about ranking samples.
'''
def rank_mean(probas):
    """
    Create mean predictions using ranks
    :param probas: 2-d array of probability values
    :return: mean ranks
    """
    ranked = []
    for i in range(probas.shape[1]):
        rank_data = stats.rankdata(probas[:, i])
        ranked.append(rank_data)
    ranked = np.column_stack(ranked)
    return np.mean(ranked, axis=1)