# Numpy and pandas by default assume a narrow screen - this fixes that
from fastai2.vision.all import *
from nbdev.showdoc import *
from ipywidgets import widgets

import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 200
mpl.rcParams['savefig.dpi']= 200
mpl.rcParams['font.size']=12

mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['font.family']='sans-serif' 
#解决负号'-'显示为方块的问题 
mpl.rcParams['axes.unicode_minus'] = False 



set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pd.set_option('display.max_columns',999)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

def get_image_files_sorted(path, recurse=True, folders=None): return get_image_files(path, recurse, folders).sorted()

# +
# pip install azure-cognitiveservices-search-imagesearch

from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth

def search_images_bing(key, term, min_sz=128):
    client = api('https://api.cognitive.microsoft.com', auth(key))
    return L(client.images.search(query=term, count=150, min_height=min_sz, min_width=min_sz).value)


# -
def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)

# heatmap method @Ebby
def show_heatmap(im,learn,cat=2):
    m = learn.model.eval()
    xb, = first(learn.dls.test_dl([im]))
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    
    x_dec = TensorImage(learn.dls.train.decode((xb,))[0][0])
    _,ax = plt.subplots()
    x_dec.show(ctx=ax)
    ax.imshow(avg_acts, alpha=0.6, extent=(0,xb.shape[2],xb.shape[3],0),
                  interpolation='bilinear', cmap='magma');