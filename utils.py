# Numpy and pandas by default assume a narrow screen - this fixes that
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastai.data.core import Datasets

# Bing Downloader
from bing_image_downloader import downloader

from nbdev.showdoc import *
from ipywidgets import widgets, Layout, IntSlider

import os

import bqplot
import bqplot.pyplot as bqpyplot
import pandas as pd

import numpy as np

# Configs for matplotlib
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
from itertools import chain

# A new method for search_images_bing
def search_images_bing(term, total_count=150, min_sz=128, key=''):
    """Search for images using the Bing API
    
    :param key: Your Bing API key
    :type key: str
    :param term: The search term to search for
    :type term: str
    :param total_count: The total number of images you want to return (default is 150)
    :type total_count: int
    :param min_sz: the minimum height and width of the images to search for (default is 128)
    :type min_sz: int
    :returns: An L-collection of ImageObject
    :rtype: L
    """
    max_count = 150
    client = api("https://api.cognitive.microsoft.com", auth(key))
    imgs = [
        client.images.search(
            query=term, min_height=min_sz, min_width=min_sz, count=count, offset=offset
        ).value
        for count, offset in (
            (
                max_count if total_count - offset > max_count else total_count - offset,
                offset,
            )
            for offset in range(0, total_count, max_count)
        )
    ]
    return L(chain(*imgs)).attrgot('content_url')

def download_datasets(labels,imgs_dir="data",max_n=150):
    path = Path(imgs_dir)

    if not path.exists():
        path.mkdir()
    for o in labels:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        print(f"===>正在下载:{o}")
        results = search_images_bing(f'{o}',total_count=max_n)
        download_images(dest, urls=results)
    return path
        
# data clean
def data_clean(images):
    failed = verify_images(images)
    failed_p = [str(o) + "\n" for o in failed] 
    l = "".join(failed_p)
    print(f">当前数据集中存在{len(failed)}张异常数据;\n------------------------------\n {l}")
    failed.map(Path.unlink)

# -
def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)

# heatmap method @Ebby
def show_heatmap(im,learn):
    result, cat, probs = learn.predict(im)
    m = learn.model.eval()
    xb, = first(learn.dls.test_dl([im]))
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,cat].backward()
    
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    
    x_dec = TensorImage(learn.dls.train.decode((xb,))[0][0])
    _,ax = plt.subplots()
    x_dec.show(ctx=ax)
    ax.set_title(f'预测结果：{result}({probs.max()*100:.2f}%)')
    ax.imshow(avg_acts, alpha=0.5, extent=(0,xb.shape[2],xb.shape[3],0),
                  interpolation='bilinear', cmap='magma');
    
    
class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()
        
class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()
        
# heatmap method @Ebby
def show_heatmap_GCAM(im,learn,cat=2):
    m = learn.model.eval()
    xb, = first(learn.dls.test_dl([im]))
    with HookBwd(learn.model[0]) as hook_g: 
        with Hook(learn.model[0]) as hook:
            preds = m(xb)
            act = hook.stored
        preds[0,int(cat)].backward()
        grad = hook_g.stored
    
    w = grad[0].mean(dim=[1,2], keepdim=True)
    cam_map = (w * act[0]).sum(0)
    
    x_dec = TensorImage(learn.dls.train.decode((xb,))[0][0])
    _,ax = plt.subplots()
    x_dec.show(ctx=ax)
    ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,xb.shape[2],xb.shape[3],0),
                  interpolation='bilinear', cmap='magma');
    
    
class ResultsWidget(object):
    IM_WIDTH = 500  # pixels

    def __init__(self, dataset: Datasets, y_score: np.ndarray, y_label: iter):
        """Helper class to draw and update Image classification results widgets.

        Args:
            dataset (Datasets): Data used for prediction, containing ImageList x and CategoryList y.
            y_score (np.ndarray): Predicted scores.
            y_label (iterable): Predicted labels. Note, not a true label.
        """
        assert len(y_score) == len(y_label) == len(dataset)

        self.dataset = dataset
        self.pred_scores = y_score
        self.pred_labels = y_label

        # Init
        self.vis_image_index = 0
        self.labels = dataset.vocab
        self.label_to_id = {s: i for i, s in enumerate(self.labels)}

        self._create_ui()

    @staticmethod
    def _list_sort(list1d, reverse=False, comparison_fn=lambda x: x):
        """Sorts list1f and returns (sorted list, list of indices)"""
        indices = list(range(len(list1d)))
        tmp = sorted(zip(list1d, indices), key=comparison_fn, reverse=reverse)
        return list(map(list, list(zip(*tmp))))

    def show(self):
        return self.ui

    def update(self):
        scores = self.pred_scores[self.vis_image_index]
        im = self.dataset[self.vis_image_index][0]  # fastai Image object

        _, sort_order = self._list_sort(scores, reverse=True)
        pred_labels_str = ""
        for i in sort_order:
            pred_labels_str += f"{self.labels[i]} ({scores[i]:3.2f})\n"
        self.w_pred_labels.value = str(pred_labels_str)

        self.w_image_header.value = f"Image index: {self.vis_image_index}"

        self.w_img.value = im._repr_png_()
        # Fix the width of the image widget and adjust the height
        self.w_img.layout.height = (
            f"{int(self.IM_WIDTH * (im.size[1]/im.size[0]))}px"
        )

        self.w_gt_label.value = str(self.labels[self.dataset[self.vis_image_index][1]])

        self.w_filename.value = str(
            self.dataset.items[self.vis_image_index].name
        )
        self.w_path.value = str(
            self.dataset.items[self.vis_image_index].parent
        )
        bqpyplot.clear()
        bqpyplot.bar(
            self.labels,
            scores,
            align="center",
            alpha=1.0,
            color=np.abs(scores),
            scales={"color": bqplot.ColorScale(scheme="Blues", min=0)},
        )

    def _create_ui(self):
        """Create and initialize widgets"""
        # ------------
        # Callbacks + logic
        # ------------
        def image_passes_filters(image_index):
            """Return if image should be shown."""
            actual_label = int(self.dataset[image_index][1])
            bo_pred_correct = bool(actual_label == self.label_to_id[self.pred_labels[image_index]])
            if (bo_pred_correct and self.w_filter_correct.value) or (
                not bo_pred_correct and self.w_filter_wrong.value
            ):
                return True
            return False

        def button_pressed(obj):
            """Next / previous image button callback."""
            step = int(obj.value)
            self.vis_image_index += step
            self.vis_image_index = min(
                max(0, self.vis_image_index), int(len(self.pred_labels)) - 1
            )
            while not image_passes_filters(self.vis_image_index):
                self.vis_image_index += step
                if (
                    self.vis_image_index <= 0
                    or self.vis_image_index >= int(len(self.pred_labels)) - 1
                ):
                    break
            self.vis_image_index = min(
                max(0, self.vis_image_index), int(len(self.pred_labels)) - 1
            )
            self.w_image_slider.value = self.vis_image_index
            self.update()

        def slider_changed(obj):
            """Image slider callback.
            Need to wrap in try statement to avoid errors when slider value is not a number.
            """
            try:
                self.vis_image_index = int(obj["new"]["value"])
                self.update()
            except Exception:
                pass

        # ------------
        # UI - image + controls (left side)
        # ------------
        w_next_image_button = widgets.Button(description="下一张")
        w_next_image_button.value = "1"
        w_next_image_button.layout = Layout(width="80px")
        w_next_image_button.on_click(button_pressed)
        w_previous_image_button = widgets.Button(description="上一张")
        w_previous_image_button.value = "-1"
        w_previous_image_button.layout = Layout(width="80px")
        w_previous_image_button.on_click(button_pressed)

        self.w_filename = widgets.Text(
            value="", description="名称:", layout=Layout(width="200px")
        )
        self.w_path = widgets.Text(
            value="", description="路径:", layout=Layout(width="200px")
        )

        self.w_image_slider = IntSlider(
            min=0,
            max=len(self.pred_labels) - 1,
            step=1,
            value=self.vis_image_index,
            continuous_update=False,
        )
        self.w_image_slider.observe(slider_changed)
        self.w_image_header = widgets.Text("", layout=Layout(width="130px"))
        self.w_img = widgets.Image()
        self.w_img.layout.width = f"{self.IM_WIDTH}px"
        w_header = widgets.HBox(
            children=[
                w_previous_image_button,
                w_next_image_button,
                self.w_image_slider,
                self.w_filename,
                self.w_path,
            ]
        )

        # ------------
        # UI - info (right side)
        # ------------
        w_filter_header = widgets.HTML(
            value="Filters (use Image +1/-1 buttons for navigation):"
        )
        self.w_filter_correct = widgets.Checkbox(
            value=True, description="正确分类图像"
        )
        self.w_filter_wrong = widgets.Checkbox(
            value=True, description="错误分类图像"
        )

        w_gt_header = widgets.HTML(value="真实值:")
        self.w_gt_label = widgets.Text(value="")
        self.w_gt_label.layout.width = "360px"

        w_pred_header = widgets.HTML(value="预测值:")
        self.w_pred_labels = widgets.Textarea(value="")
        self.w_pred_labels.layout.height = "200px"
        self.w_pred_labels.layout.width = "360px"

        w_scores_header = widgets.HTML(value="分类得分:")
        self.w_scores = bqpyplot.figure()
        self.w_scores.layout.height = "250px"
        self.w_scores.layout.width = "370px"
        self.w_scores.fig_margin = {
            "top": 5,
            "bottom": 80,
            "left": 30,
            "right": 5,
        }

        # Combine UIs into tab widget
        w_info = widgets.VBox(
            children=[
                w_filter_header,
                self.w_filter_correct,
                self.w_filter_wrong,
                w_gt_header,
                self.w_gt_label,
                w_pred_header,
                self.w_pred_labels,
                w_scores_header,
                self.w_scores,
            ]
        )
        w_info.layout.padding = "20px"
        self.ui = widgets.Tab(
            children=[
                widgets.VBox(
                    children=[
                        w_header,
                        widgets.HBox(children=[self.w_img, w_info]),
                    ]
                )
            ]
        )
        self.ui.set_title(0, "模型结果分析系统")

        # Fill UI with content
        self.update()


def show_all_test_imgs(learn,interp):
    pred_scores = to_np(interp.preds)
    w_results = ResultsWidget(
        dataset=learn.dls.valid_ds,
        y_score=pred_scores,
        y_label=[learn.dls.vocab[x] for x in np.argmax(pred_scores, axis=1)],
    )
    display(w_results.show())        


'''
修改Interpretation中的plot loss方法，把有些展示内容做的更加直观

interpret中核心函数是通过 self.dl_pre_show_batch(b) 实现信息的解码
'''
from random import randint
def plot_results(interp:ClassificationInterpretation, n=4):
    losses,idx = interp.top_losses()
    max_n = len(idx)
    if not isinstance(interp.inputs, tuple): interp.inputs = (interp.inputs,)
    if isinstance(interp.inputs[0], Tensor): inps = tuple(o[idx] for o in interp.inputs)
    else: inps = interp.dl.create_batch(interp.dl.before_batch([tuple(o[i] for o in interp.inputs) for i in idx]))
        
    b = inps + tuple(o[idx] for o in (interp.targs if is_listy(interp.targs) else (interp.targs,)))
    x,y,its = interp.dl._pre_show_batch(b,max_n)
    
    b_out = inps + tuple(o[idx] for o in (interp.decoded if is_listy(interp.decoded) else (interp.decoded,)))
    x1,y1,outs = interp.dl._pre_show_batch(b_out, max_n)
    preds = outs.itemgot(slice(len(inps), None))
    
    fig, ax = plt.subplots(3,n, figsize=(20,15))
    fig.suptitle('预测结果分析(预测值/真实值/损失/置信度)',fontsize=20)
    # Random
    for i in range(n):
        r_index = randint(0,max_n)
        id = idx[r_index]
        (im, ture_v) = its[r_index] 
        im = im.permute((1,2,0))
        ax[0,i].imshow(im)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(f'{preds[r_index][0]} / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[r_index]]:.2f}')
    ax[0,0].set_ylabel('随机\n取样', fontsize=16, rotation=0, labelpad=80)
    
    
    for i in range(n):
        id = idx[i]
        #im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]
        (im, ture_v) = its[i] 
        im = im.permute((1,2,0))
        ax[1,i].imshow(im)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].set_title(f'{preds[i][0]}  / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[i]]:.2f}')
    ax[1,0].set_ylabel('最差\n预测', fontsize=16, rotation=0, labelpad=80)


    # Most correct or least losses
    for i in range(n):
        t = max_n - i - 1
        id = idx[t]
        (im, ture_v) = its[t] 
        im = im.permute((1,2,0))
        ax[2,i].imshow(im)
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].set_title(f'{preds[t][0]}  / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[t]]:.2f}')
    ax[2,0].set_ylabel('最佳\n预测', fontsize=16, rotation=0, labelpad=80)
    
    
    '''
修改Interpretation中的plot loss方法，把有些展示内容做的更加直观

interpret中核心函数是通过 self.dl_pre_show_batch(b) 实现信息的解码
'''
    
# hook into forward pass
def hooked_backward(m, oneBatch, cat):
    # we hook into the convolutional part = m[0] of the model
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(oneBatch)
            preds[0,int(cat)].backward()
    return hook_a,hook_g

# We can create a utility function for getting a validation image with an activation map
def getHeatmap(learner, input, target):
    """Returns the validation set image and the activation map"""
    # this gets the model
    m = learner.model.eval().cpu()   

    # attach hooks
    hook_a,hook_g = hooked_backward(m, input, target)
    # get convolutional activations and average from channels
    acts = hook_a.stored[0].cpu()
    #avg_acts = acts.mean(0)

    # Grad-CAM
    grad = hook_g.stored[0][0].cpu()
    grad_chan = grad.mean(1).mean(1)
    grad.shape,grad_chan.shape
    mult = (acts*grad_chan[...,None,None]).mean(0)
    return mult
    
# Then, modify our plotting func a bit
def plot_heatmap_overview(interp:ClassificationInterpretation,learn, n=4):
    # top losses will return all validation losses and indexes sorted by the largest first
 
    losses,idx = interp.top_losses()
    max_n = len(idx)
    if not isinstance(interp.inputs, tuple): interp.inputs = (interp.inputs,)
    if isinstance(interp.inputs[0], Tensor): inps = tuple(o[idx] for o in interp.inputs)
    else: inps = interp.dl.create_batch(interp.dl.before_batch([tuple(o[i] for o in interp.inputs) for i in idx]))
        
    b = inps + tuple(o[idx] for o in (interp.targs if is_listy(interp.targs) else (interp.targs,)))
    x,y,its = interp.dl._pre_show_batch(b,max_n)
    
    b_out = inps + tuple(o[idx] for o in (interp.decoded if is_listy(interp.decoded) else (interp.decoded,)))
    x1,y1,outs = interp.dl._pre_show_batch(b_out, max_n)
    preds = outs.itemgot(slice(len(inps), None))
    
    
    
    fig, ax = plt.subplots(3,n, figsize=(20,16))
    fig.suptitle('模型预测分析(预测值/实际值/损失/置信度)',fontsize=20)

    # Random
    for i in range(n):
        r_index = randint(0,max_n)
        id = idx[r_index]
        (im, ture_v) = its[r_index] 
        im = im.permute((1,2,0))
        act = getHeatmap(learn, interp.inputs[0][id].unsqueeze(0), interp.targs[id])
        
        H,W = im.shape[:2]
 
        ax[0,i].imshow(im)
        ax[0,i].imshow(im, cmap=plt.cm.gray)
        ax[0,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
              interpolation='bilinear', cmap='inferno')
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        ax[0,i].set_title(f'{preds[r_index][0]} / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[r_index]]:.2f}')
    ax[0,0].set_ylabel('随机\n取样', fontsize=16, rotation=0, labelpad=80)
      
    # Most incorrect or top losses
    for i in range(n):
        id = idx[i]
        act = getHeatmap(learn, interp.inputs[0][id].unsqueeze(0), interp.targs[id])
        (im, ture_v) = its[i] 
        im = im.permute((1,2,0))
        
        H,W = im.shape[:2]

        ax[1,i].imshow(im)
        ax[1,i].imshow(im, cmap=plt.cm.gray)
        ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
              interpolation='bilinear', cmap='inferno')
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
        ax[1,i].set_title(f'{preds[i][0]}  / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[i]]:.2f}')
    ax[1,0].set_ylabel('最差\n预测', fontsize=16, rotation=0, labelpad=80)
    
    
    # Most correct or least losses
    for i in range(n):
        t = max_n - i - 1
        id = idx[t]
        act = getHeatmap(learn, interp.inputs[0][id].unsqueeze(0), interp.targs[id])
        (im, ture_v) = its[t] 
        im = im.permute((1,2,0))
        
        H,W = im.shape[:2]
        ax[2,i].imshow(im)
        ax[2,i].imshow(im, cmap=plt.cm.gray)
        ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),
              interpolation='bilinear', cmap='inferno')
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
        ax[2,i].set_title(f'{preds[t][0]}  / {ture_v} / {interp.losses[id]:.2f} / {interp.preds[id][y1[t]]:.2f}')
    ax[2,0].set_ylabel('最佳\n预测', fontsize=16, rotation=0, labelpad=80)    
    
    
from matplotlib.axes import Axes      
from matplotlib.text import Annotation

def add_value_labels(
    ax: Axes, spacing: int = 5, percentage: bool = False
) -> None:
    """ Add labels to the end of each bar in a bar chart.

    Overwrite labels on axes if they already exist.

    Args:
        ax (Axes): The matplotlib object containing the axes of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        percentage (bool): if y-value is a percentage
    """
    for child in ax.get_children():
        if isinstance(child, Annotation):
            child.remove()

    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        label = (
            "{:.2f}%".format(y_value * 100)
            if percentage
            else "{:.1f}".format(y_value)
        )

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha="center",  # Horizontally center label
            va="bottom",  # Vertically align label
        )