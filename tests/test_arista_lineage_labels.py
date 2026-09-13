from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from reproduction.arista.simulate_paper_populations import classify_lineage


def test_lineage_keeps_classifier_predictions_despite_neighbour_majority():
    class Classifier(torch.nn.Module):
        def forward(self, x):
            return torch.stack((-x[:, -1], x[:, -1]), dim=1)
    classifier = SimpleNamespace(model=Classifier(), feature_dim=3,
        include_time_feature=True, label_encoder=LabelEncoder().fit(['MCG', 'reaEGC']))
    # All cells share a position. The rare reaEGC prediction must not be voted away.
    states = np.zeros((2, 12, 3), dtype=np.float32)
    states[:, :, 2] = -10
    states[:, 0, 2] = 10
    labels = classify_lineage(states, [0., 1.], classifier, 'cpu')
    for values in labels:
        assert values[0] == 'reaEGC'
        assert np.count_nonzero(values == 'MCG') == 11
