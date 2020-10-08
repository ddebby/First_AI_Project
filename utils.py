# Numpy and pandas by default assume a narrow screen - this fixes that
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastai.data.core import Datasets


# from fastai2.vision.all import *
# from fastai2.vision.widgets import *
# from fastai2.data.core import Datasets
from nbdev.showdoc import *
from ipywidgets import widgets, Layout, IntSlider

import os

import bqplot
import bqplot.pyplot as bqpyplot
import pandas as pd

import numpy as np

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
def search_images_bing(key, term, total_count=150, min_sz=128):
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
    return L(chain(*imgs))


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