import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

def index_iter(n_obs, batch_size):
    indices = np.random.permutation(n_obs)
    for i in range(0, n_obs, batch_size):
        yield indices[i: min(i + batch_size, n_obs)]

def train_classifier(adata, obs_key, classifier, lr, batch_size, num_epochs,
                      test_adata=None, optim=torch.optim.Adam, accuracy=True, **kwargs):

    optimizer = optim(classifier.parameters(), lr=lr, **kwargs)
    criterion = torch.nn.CrossEntropyLoss()

    le = LabelEncoder()
    le.fit(np.unique(adata.obs[obs_key]))

    if test_adata is None:
        t_X = torch.Tensor(adata.X)
        t_labels = torch.LongTensor(le.transform(adata.obs[obs_key]))
        comment = '-- total train accuracy: ' if accuracy else  '-- total train loss:'
    else:
        t_X = torch.Tensor(test_adata.X)
        t_labels = torch.LongTensor(le.transform(test_adata.obs[obs_key]))
        comment = '-- test accuracy:' if accuracy else  '-- test loss:'

    for epoch in range(num_epochs):
        print('Epoch:', epoch)

        classifier.train()

        for step, selection in enumerate(index_iter(adata.n_obs, batch_size)):
            batch = torch.Tensor(adata.X[selection])
            labels = torch.LongTensor(le.transform(adata.obs[obs_key][selection]))

            optimizer.zero_grad()

            out = classifier(batch)
            loss = criterion(out, labels)

            loss.backward()

            optimizer.step()

            if step % 100 == 0:
                print('Step:', step, '| batch train loss: %.4f' % loss.data.numpy())

        classifier.eval()

        if accuracy:
            out = classifier.predict(t_X)
            loss = out.eq(t_labels).sum() / float(t_X.shape[0]) * 100
        else:
            out = classifier(t_X)
            loss = criterion(out, t_labels)
        print('Epoch:', epoch, comment, '%.4f' % loss.data.numpy())
