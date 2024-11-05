import time
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
torch.set_grad_enabled(False)
images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'


extractor = SuperPoint(max_num_keypoints=50).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)


image0 = load_image(images / "perception1.jpg")
image1 = load_image(images / "perception2.jpg")


superpoint_times = []
lightglue_times = []

run_times=20

for i in range(run_times):
    start_time = time.time()
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    end_time = time.time()
    superpoint_times.append(end_time - start_time)


    start_time = time.time()
    matches01 = matcher({"image0": feats0, "image1": feats1})
    end_time = time.time()
    lightglue_times.append(end_time - start_time)


avg_superpoint_time = sum(superpoint_times) / len(superpoint_times)
avg_lightglue_time = sum(lightglue_times) / len(lightglue_times)
superpoint_fps = 1 / avg_superpoint_time
lightglue_fps = 1 / avg_lightglue_time

print(f"SuperPoint average latency: {avg_superpoint_time:.4f} s, FPS: {superpoint_fps:.2f}")
print(f"LightGlue average latency: {avg_lightglue_time:.4f} s, FPS: {lightglue_fps:.2f}")


feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]


kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
matches = matches01["matches"]


matched_points = []
for match in matches:
    idx0, idx1 = match[0], match[1]
    point0 = kpts0[idx0].tolist()
    point1 = kpts1[idx1].tolist()
    matched_points.append((point0, point1))


for i, (pt0, pt1) in enumerate(matched_points):
    print(f"match pairs {i + 1}: image1 {pt0} <--> image2 {pt1}")


m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)


kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)


plt.show()
