from PIL import Image
import os
import argparse

# running the preprocessing

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--folder_name', type=str, default="", required=True)
    opt = parser.parse_args()
    return opt

def resize_img(path):
    im = Image.open(path)
    im = im.resize((768, 1024))
    im.save(path)

opt = get_opt()


for path in os.listdir(f'/content/inputs_{opt.folder_name}/test/cloth/'):
    resize_img(f'/content/inputs_{opt.folder_name}/test/cloth/{path}')

os.chdir('/content/clothes-virtual-try-on')
os.system(f"rm -rf /content/inputs_{opt.folder_name}/test/cloth/.ipynb_checkpoints")
os.system(f"python cloth-mask.py --folder_name {opt.folder_name}")
os.chdir('/content')
os.system(f"python /content/clothes-virtual-try-on/remove_bg.py --folder_name {opt.folder_name}")
os.system(
    f"python3 /content/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'lip' --model-restore '/content/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/content/inputs_{opt.folder_name}/test/image' --output-dir '/content/inputs_{opt.folder_name}/test/image-parse'")
os.chdir('/content')
os.system(
    f"cd openpose && ./build/examples/openpose/openpose.bin --image_dir /content/inputs_{opt.folder_name}/test/image/ --write_json /content/inputs_{opt.folder_name}/test/openpose-json/ --display 0 --render_pose 0 --hand")
os.system(
    f"cd openpose && ./build/examples/openpose/openpose.bin --image_dir /content/inputs_{opt.folder_name}/test/image/ --display 0 --write_images /content/inputs_{opt.folder_name}/test/openpose-img/ --hand --render_pose 1 --disable_blending true")

model_image = os.listdir(f'/content/inputs_{opt.folder_name}/test/image')
cloth_image = os.listdir(f'/content/inputs_{opt.folder_name}/test/cloth')
pairs = zip(model_image, cloth_image)

with open(f'/content/inputs_{opt.folder_name}/test_pairs.txt', 'w') as file:
    for model, cloth in pairs:
        file.write(f"{model} {cloth}")

# making predictions
os.system(
    f"python /content/clothes-virtual-try-on/test.py --name output_{opt.folder_name} --dataset_dir /content/inputs_{opt.folder_name} --checkpoint_dir /content/clothes-virtual-try-on/checkpoints --save_dir /content/")
os.system(f"rm -rf /content/inputs_{opt.folder_name}")
os.system(f"rm -rf /content/output_{opt.folder_name}/.ipynb_checkpoints")
