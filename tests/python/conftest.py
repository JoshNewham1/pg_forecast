import os
import sys

tfb_path = os.path.abspath(os.path.join(__file__, "../../../eval/TFB"))
monash_path = os.path.abspath(os.path.join(__file__, "../../../eval"))
if tfb_path not in sys.path:
    sys.path.insert(0, tfb_path)
if monash_path not in sys.path:
    sys.path.insert(0, monash_path)
