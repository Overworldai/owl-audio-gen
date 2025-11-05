# Owl Audio Gen Experiments

**Setup**  
```
git submodule init
git submodule update
cd owl-vaes
git switch waypoint_1_prep
pip install -r requirements.txt
```

**Loading Audio VAE on cluster**
```python
import sys
sys.path.append("./owl-vaes")
from owl_vaes import from_pretrained

cfg_path = "owl-vaes/configs/waypoint_1_audio/basic.yml"
ckpt_path = "/mnt/data/shahbuland/owl-vaes/checkpoints/waypoint_1_audio_basic/step_105000.pt"

vae = from_pretrained(cfg_path, ckpt_path)
```
