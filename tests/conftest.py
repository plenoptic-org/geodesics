import pytest
import plenoptic as po
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

torch.set_num_threads(1)  # torch uses all avail threads which will slow tests
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
class ColorModel(torch.nn.Module):
    """Simple model that takes color image as input and outputs 2d conv."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(scope='package')
def curie_img():
    return po.data.curie().to(DEVICE)


@pytest.fixture(scope='package')
def einstein_img():
    return po.data.einstein().to(DEVICE)


@pytest.fixture(scope='package')
def einstein_small_seq(einstein_img_small):
    return po.tools.translation_sequence(einstein_img_small, 5)


@pytest.fixture(scope='package')
def einstein_img_small(einstein_img):
    return po.tools.center_crop(einstein_img, 64).to(DEVICE)


@pytest.fixture(scope='package')
def color_img():
    img = po.data.color_wheel().to(DEVICE)
    return img[..., :256, :256]


def get_model(name):
    if name == 'Identity':
        return po.simul.models.naive.Identity().to(DEVICE)
    elif name == 'ColorModel':
        model = ColorModel().to(DEVICE)
        po.tools.remove_grad(model)
        return model
    # FrontEnd models:
    elif name == 'frontend.OnOff.nograd':
        mdl = po.simul.OnOff((31, 31), pretrained=True, cache_filt=True).to(DEVICE)
        po.tools.remove_grad(mdl)
        return mdl


@pytest.fixture(scope='package')
def model(request):
    return get_model(request.param)
