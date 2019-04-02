import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torch.autograd import Variable
from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.ssd import SSD512, SSDBoxCoder


print('Loading model..')
NUM_CLASSES=5
BS = 8
# Model
print('==> Building model..')
net = SSD512(num_classes=NUM_CLASSES)

checkpoint = torch.load('./model_chkps/model_chkp_21_epochs_new.pth')
net.load_state_dict(checkpoint['net'])
    

#net.load_state_dict(torch.load('./model_chkps/model_chkp.pth'),strict=False)
net.eval()

print('Loading image..')
IMAGE_PATHS = [
 '/data/deva/recaptcha_data/labelled/pants/beige_AntiqueWhite_64_3002_310094533_1712914711_Men_Pants_Jeans.jpg',
 '/data/deva/recaptcha_data/labelled/shorts/beige_AntiqueWhite_70_3020_2325289464_12214026386_Men_Shorts.jpg',
 '/data/deva/recaptcha_data/unlabelled/beige_Ecru_21_3002_1752206130_12178447290_Men_Pants_Jeans.jpg',
 '/data/deva/recaptcha_data/labelled/pants/blue_DarkSlateGray_46_3002_2846525030_12090766970_Men_Pants_Jeans.jpg',
 '/data/deva/recaptcha_data/unlabelled/beige_Ecru_23_3075_1704375222_9495561906_Men_Shirts.jpg',
 '/data/deva/recaptcha_data/unlabelled/black_Black_65_3002_529925020_4276264527_Men_Pants_Jeans.jpg',
]

for image_path in IMAGE_PATHS:

    img = Image.open(image_path)

    ow = oh = 512
    img = img.resize((ow,oh))

    print('Predicting..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    x = transform(img)
    x = Variable(x, volatile=True)
    loc_preds, cls_preds = net(x.unsqueeze(0))
    
    print('Decoding..')
    box_coder = SSDBoxCoder(net)
    boxes, labels, scores = box_coder.decode(
        loc_preds.data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).data)
    print(labels)
    print(scores)
    print(boxes)
    
    draw = ImageDraw.Draw(img)
    
    L = ["pants","shirt","shorts","tshirt"]
    
    for box,label,score in zip(boxes,labels,scores):
        x1,y1,x2,y2 = list(box)
        draw.rectangle(list(box), outline='red')
        draw.rectangle((x1+1,y1+1,x1+150,y1+10), fill='black')
        draw.text((x1+3,y1+1),"{} {:0.2f}".format(L[label],score), fill='green')

    img.show()
