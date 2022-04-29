"""图像质量自动评价
"""

from jindai import Plugin, expand_path
from jindai.models import ImageItem
from plugins.gallery import ImageOrAlbumStage

MODEL = expand_path('models_data/nima.pkl')
try:
    import torch
except ImportError as e:
    print("Please install pytorch first.")
    raise e


class NIMAEval(ImageOrAlbumStage):
    """图像质量自动评价
    """

    def __init__(self):
        from .adapter import load_state as nima_init, predict as nima_predict
        nima_init(MODEL)
        super().__init__()
        self.predict = nima_predict

    def resolve_image(self, i: ImageItem, _):
        for (_, mean) in self.predict([i.image]):
            i.ava_eval = mean
            i.save()


class NIMAPlugin(Plugin):
    """图像质量评价插件"""

    def __init__(self, app, **_):
        super().__init__(app)
        ImageItem.set_field('ava_eval', float)
        self.register_pipelines(globals())
