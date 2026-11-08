"""Run Oura's NSSA sleep stager (sleepstaging_2_6_0, the model behind the official
app's hypnogram) on inputs from tools/nssa_inputs.py.

The TorchScript graph carries all of Oura's pre/post-processing (feature
extraction, smoothing, wake re-run, bedtime grid, false-nap check). It calls five
native `oura_ops` from libalgos.so; tools/nssa_ops.py registers our transcriptions
of them, including the tree-ensemble classifier (tools/nssa_classifier.py).
"""
import torch

import nssa_ops  # noqa: F401  (registers oura_ops::*)

MODEL_NAME = "sleepstaging_2_6_0.pt"
SEX_PYTORCH = {"F": -1.0, "M": 1.0}  # DbUserSettingsUtils.getPytorchValue; other = 0


def demographics(age, height_m, weight_kg, sex):
    """NssaPyTorchModel demographics vector [age, bmi, sex] (float32). Age is an
    integer number of years and BMI = weight / height**2 (DbUserSettingsUtils)."""
    bmi = float(weight_kg) / (float(height_m) ** 2)
    return [float(int(age)), bmi, SEX_PYTORCH.get(str(sex).upper()[:1], 0.0)]


class NssaStager:
    def __init__(self, model_path):
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()

    def stage(self, inputs, demo):
        """inputs: collect_nssa_inputs(...) dict. Returns (epoch_start_ms list, codes
        list (1=DEEP 2=LIGHT 3=REM 4=WAKE)) or raises ValueError when the model's own
        input validator would reject the night (the app then has no hypnogram)."""
        acm, ibi, temp = inputs["acm"], inputs["ibi"], inputs["temp"]
        if not acm or not ibi or not temp:
            raise ValueError("NSSA needs sleep ACM, IBI and sleep temperature inside the bedtime")
        acm_ts = torch.tensor([a[0] for a in acm], dtype=torch.int64)
        acm_val = torch.tensor([a[1] for a in acm], dtype=torch.float32)
        ibi_ts = torch.tensor([b[0] for b in ibi], dtype=torch.int64)
        ibi_val = torch.tensor([[b[1], b[2], b[3]] for b in ibi], dtype=torch.float32)
        temp_ts = torch.tensor([t[0] for t in temp], dtype=torch.int64)
        temp_val = torch.tensor([t[1] for t in temp], dtype=torch.float32)
        scalars = torch.tensor(demo, dtype=torch.float32)
        bedtime = torch.tensor(list(inputs["bedtime"]), dtype=torch.int64)
        with torch.no_grad():
            ts, stages, _features, _debug = self.model(
                acm_val, acm_ts, ibi_val, ibi_ts, temp_val, temp_ts, scalars, bedtime)
        codes = [int(round(float(c))) for c in stages[:, 0].tolist()]
        return [int(t) for t in ts.reshape(-1).tolist()], codes
