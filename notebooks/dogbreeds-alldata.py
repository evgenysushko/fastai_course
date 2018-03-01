from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = '../data/dogbreeds/'
sz = 224
arch = resnext101_64
bs = 58

def get_data(sz,bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv',
                                        test_name='test', val_idxs=[0],
                                        suffix='.jpg', tfms=tfms, bs=bs)
    return data if sz > 300 else data.resize(340, 'tmp')

data = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)
learn.fit(1e-1, 2)
print('precompute trained')
learn.precompute=False
learn.fit(1e-1, 5, cycle_len=1)
learn.save('224_pre_all')
print('224 trained')

learn.set_data(get_data(299, bs))
learn.fit(1e-2, 3, cycle_len=1)
learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2)
learn.fit(1e-2, 2, cycle_len=2)
learn.save('229_pre_all')
print('229 trained')
#learn.set_data(get_data(350, bs))
#learn.fit(1e-2, 2, cycle_len=2)
#learn.save('350_pre_all')

test = learn.predict(is_test=True)
print('prediction done')

columns = pd.read_csv(f'{PATH}sample_submission.csv', index_col='id').columns
test = pd.DataFrame(np.exp(test))
test.columns = columns
test.index = [i.split('.jpg')[0].split('/')[-1] for i in data.test_dl.dataset.fnames]
test.index.name ='id'
test.to_csv(f'{PATH}submission_all.csv')

#!kg submit {PATH}submission_all.csv -m 'all data'
