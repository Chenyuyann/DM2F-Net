import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage
from model import DM2FNet, DM2FNet_woPhy

ckpt_path = './ckpt'
exp_name = 'RESIDE_ITS'
# exp_name = 'O-Haze'

args = {
    # RESIDE
    'snapshot': 'iter_30000_loss_0.02194_lr_0.000144',

    # O-Haze
    # 'snapshot': 'iter_20000_loss_0.05028_lr_0.000000',
}

def load_and_process_image(image_path, net):
    img = Image.open(image_path).convert('RGB')
    input_tensor = ToTensor()(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output_tensor = net(input_tensor).detach()

    output_image = ToPILImage()(output_tensor.squeeze())
    return output_image

def main():
    if 'O-Haze' in exp_name:
        net = DM2FNet_woPhy().cuda()
    else:
        net = DM2FNet().cuda()
    if len(args['snapshot']) > 0:
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
    net.eval()

    image_paths = [
        'data/Others/berlin-8429763_1920.jpg',
        'data/Others/budapest-2058395_1920.jpg',
        'data/Others/city-5721873_1920.jpg',
        'data/Others/tokyo-1141165_1920.jpg',
        'data/Others/bridge-2617838_1920.jpg',
        'data/Others/church-473735_1920.jpg',
        'data/Others/dom-8429773_1920.jpg',
        'data/Others/moutains-3974479_1920.jpg',
        'data/Others/castle-5734731_1920.jpg',
        'data/Others/city-5184919_1920.jpg',
        'data/Others/haze-5184916_1920.jpg',
        'data/Others/niagara-falls-4083_1920.jpg',
        'data/Others/city-5929528_1920.jpg',
        'data/Others/rostock-3965049_1920.jpg',
        'data/Others/turku-577286_1920.jpg',
    ]

    output_dir = os.path.join('ckpt/Others', '%s_%s' % (exp_name, args['snapshot']))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in image_paths:
        output_image = load_and_process_image(image_path, net)
        file_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, file_name)
        output_image.save(save_path)

if __name__ == '__main__':
    main()
